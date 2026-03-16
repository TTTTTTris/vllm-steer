#%%
import json
import os
import re
from functools import partial
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# STEER_VEC_PATH = "/home/jiayi_tian/TensorRouter/TensorRouter/vector_500_500/DeepSeek-R1-Distill-Qwen-1.5B/layer_21_highrank_60_transition_reflection_steervec.pt"
STEER_VEC_PATH = "/home/jiayi_tian/TensorRouter/TensorRouter/vector_500_500/DeepSeek-R1-Distill-Qwen-1.5B/layer_20_transition_reflection_steervec.pt"
TARGET_LAYER = 20
STEER_SCALE = -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Define the suffix for newline tokens in the tokenizer
target_suffix = "ĊĊ"  # "\n\n" is tokenized as "ĊĊ"

# Get complete tokenizer vocabulary
vocab = tokenizer.get_vocab()

# Find all tokens and their IDs that end with the target suffix
# These are the newline tokens we'll apply steering to
matching_tokens_ids = [
    token_id
    for token, token_id in vocab.items()
    if isinstance(token, str) and token.endswith(target_suffix)
]
MATCH_TOKEN_IDS: list[int] = matching_tokens_ids

TASK='AIME'

OUTPUT_FILE = f"results/${TASK}_static_steer_results_{STEER_SCALE}.jsonl"
INSPECT_FILE = f"results/${TASK}_static_steer_inspect_{STEER_SCALE}.json"

os.environ["HF_HOME"] = "/data/jiayi"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def build_prompt(problem: str) -> str:
    return (
        "Solve the following math problem carefully.\n"
        "Think step by step, and put the final answer inside \\boxed{}.\n\n"
        f"Problem:\n{problem}\n"
    )


def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def _enable_static_steering_worker(
    worker,
    steer_vec_path: str,
    layer_idx: int,
    scale: float,
    match_token_ids: list[int] | None,
):
    # This runs inside each vLLM worker process.
    model = worker.model_runner.model

    steer_vec = torch.load(steer_vec_path, map_location="cpu")

    steer_vec = steer_vec.float().view(-1)
    if steer_vec.numel() == 0 or torch.all(steer_vec == 0): raise ValueError(f"Invalid steer_vec from {steer_vec_path}: empty or all zeros")

    # Your custom patched API on the underlying HF-style model.
    model.model.set_static_steering(
        layer_idx=layer_idx,
        steer_vec=steer_vec,
        scale=scale,
        match_token_ids=match_token_ids,
    )
    return True


def _disable_static_steering_worker(worker):
    model = worker.model_runner.model
    model.model.disable_static_steering()
    return True


def enable_static_steering(
    llm: LLM,
    steer_vec_path: str,
    layer_idx: int,
    scale: float = 1.0,
    match_token_ids: list[int] | None = None,
):
    return llm.collective_rpc(
        _enable_static_steering_worker,
        args=(steer_vec_path, layer_idx, scale, match_token_ids),
    )


def disable_static_steering(llm: LLM):
    return llm.collective_rpc(_disable_static_steering_worker)


def _inspect_static_steering_worker(worker, layer_idx: int):
    model = worker.model_runner.model
    layer = model.model.layers[layer_idx]
    return {
        "mask": float(layer.steer_mask.item()),
        "scale": float(layer.steer_scale.item()),
        "shape": tuple(layer.steer_vec.shape),
        "dtype": str(layer.steer_vec.dtype),
        "device": str(layer.steer_vec.device),
        "norm": float(layer.steer_vec.norm().item()),
        "first8": layer.steer_vec[:8].detach().cpu().tolist(),
        "match_enabled": float(layer.steer_match_enabled.item()),
        "match_token_ids": layer.steer_match_token_ids.detach().cpu().tolist(),
    }


def inspect_static_steering(llm: LLM, layer_idx: int):
    return llm.collective_rpc(_inspect_static_steering_worker, args=(layer_idx,))


def save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

#%%
import json
from datasets import load_dataset

def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a


problems = []
answers = []

# data = load_dataset("HuggingFaceH4/MATH-500", split="test")
data = load_dataset("HuggingFaceH4/aime_2024", split="train")
for example in data:
    gt = extract_box(example["solution"])
    problems.append(example["problem"])
    answers.append(example["answer"])

# 看看前两个
print("Problems:", problems[:2])
print("Answers:", answers[:2])


examples = ["Please reason step by step, and put your final answer within \\boxed{}.\nUser: " + prompt + "\nAssistant: <think>" for prompt in problems]

llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    # no EasySteer flags needed
    # no enforce_eager needed if your static patch is CUDA-graph safe
)

# Enable your static fused steering once
enable_static_steering(
    llm,
    steer_vec_path=STEER_VEC_PATH,
    layer_idx=TARGET_LAYER,
    scale=STEER_SCALE,
    match_token_ids=MATCH_TOKEN_IDS or None,
)
# disable_static_steering(
#     llm,
# )
inspect_output = inspect_static_steering(llm, TARGET_LAYER)
print(inspect_output)
save_json(INSPECT_FILE, inspect_output)

# Generate response with SEAL steering
example_answers = llm.generate(
    examples, 
    SamplingParams(
        temperature=0,
        max_tokens=8192,
        skip_special_tokens=False,
    ), 
)

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
outputs = [output.outputs[0].text for output in example_answers]
extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
results = []
for i, llm_output in enumerate(outputs):
    gold = parse(f"${answers[i]}$", extraction_config=extraction_target)
    answer = parse(llm_output, extraction_config=extraction_target)
    result = verify(gold, answer)
    results.append(result)
accuracy = sum(results) / len(results)
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for i, llm_output in enumerate(outputs):
        file.write(json.dumps({
            "index": i,
            "problem": problems[i],
            "gold_answer": answers[i],
            "model_output": llm_output,
            "verified": bool(results[i]),
        }, ensure_ascii=False) + "\n")

save_json(OUTPUT_FILE.replace(".jsonl", "_summary.json"), {
    "steer_scale": STEER_SCALE,
    "match_token_ids": MATCH_TOKEN_IDS,
    "target_layer": TARGET_LAYER,
    "steer_vec_path": STEER_VEC_PATH,
    "accuracy": accuracy,
    "num_samples": len(results),
    "inspect_file": INSPECT_FILE,
    "results_file": OUTPUT_FILE,
})
print(accuracy)

