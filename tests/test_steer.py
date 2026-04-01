#%%
import json
import os
import re
import time
from functools import partial
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

START_TIME = time.perf_counter()


# STEER_VEC_PATH = ""
# STEER_VEC_PATH = "/home/jiayi_tian/TensorRouter/TensorRouter/vector_500_500/DeepSeek-R1-Distill-Qwen-1.5B/layer_20_transition_reflection_steervec.pt"
# STEER_VEC_PATH = "/home/jiayi/TensorRouter/TensorRouter/vector_500_500/DeepSeek-R1-Distill-Qwen-1.5B/layer_21_highrank_60_transition_reflection_steervec.pt"
# TARGET_LAYER = 20
# Read steering config from shell env (set by run.sh).
STATIC_STEER_SCALE = float(os.getenv("STATIC_STEER_SCALE", "0"))
STATIC_STEER_ENABLE = int(os.getenv("STATIC_STEER_ENABLE", "0"))
STATIC_STEER_LAYER = int(os.getenv("STATIC_STEER_LAYER", "20"))
STATIC_STEER_PATH = os.getenv("STATIC_STEER_PATH", "")
MODEL_PATH = os.getenv("MODEL_PATH")
# STATIC_STEER_DEBUG = os.getenv("STATIC_STEER_DEBUG", "0")
# STATIC_STEER_DEBUG_EVERY = os.getenv("STATIC_STEER_DEBUG_EVERY", "1")
# STATIC_STEER_DEBUG_MAX_PRINTS = os.getenv("STATIC_STEER_DEBUG_MAX_PRINTS", "500")


tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# # Define the suffix for newline tokens in the tokenizer
# target_suffix = "ĊĊ"  # "\n\n" is tokenized as "ĊĊ"

# # Get complete tokenizer vocabulary
# vocab = tokenizer.get_vocab()

# # Find all tokens and their IDs that end with the target suffix
# # These are the newline tokens we'll apply steering to
# matching_tokens_ids = [
#     token_id
#     for token, token_id in vocab.items()
#     if isinstance(token, str) and token.endswith(target_suffix)
# ]

from transformers import AutoTokenizer

print("newline:", tok.encode("\n", add_special_tokens=False))
print("double newline:", tok.encode("\n\n", add_special_tokens=False))

STATIC_STEER_MATCH_TOKEN_IDS = tok.encode("\n\n", add_special_tokens=False)[0]

NUM_SAMPLES = -1
MAX_TOKENS = 16384
task='aime_2024'

if STATIC_STEER_ENABLE==0:
    OUTPUT_FILE = f"results/{task}_serve_results_{STATIC_STEER_ENABLE}.jsonl"
else:
    OUTPUT_FILE = f"results/{task}_serve_results_{STATIC_STEER_ENABLE}_{STATIC_STEER_SCALE}_{NUM_SAMPLES}.jsonl"
print(OUTPUT_FILE)
SUMMARY_FILE = OUTPUT_FILE.replace(".jsonl", "_summary.json")
os.makedirs("results", exist_ok=True)


os.environ["HF_HOME"] = "/data/jiayi"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def build_prompt(problem: str) -> str:
    return (
        "Solve the following math problem carefully.\n"
        "Think step by step, and put the final answer inside \\boxed{}.\n\n"
        f"Problem:\n{problem}\n"
    )


def extract_boxed(text: str) -> Optional[str]:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None



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
data = load_dataset(f"HuggingFaceH4/{task}", split="train")
for example in data:
    gt = extract_box(example["solution"])
    problems.append(example["problem"])
    answers.append(example["answer"])

# 看看前两个
print("Problems:", problems[:2])
print("Answers:", answers[:2])


examples = ["Please reason step by step, and put your final answer within \\boxed{}.\nUser: " + prompt + "\nAssistant: <think>" for prompt in problems]

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    # no EasySteer flags needed
    # no enforce_eager needed if your static patch is CUDA-graph safe
)




# Generate response with SEAL steering
example_answers = llm.generate(
    examples, 
    SamplingParams(
        temperature=0,
        max_tokens=16384+2000,
        skip_special_tokens=False,
    ), 
)

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
outputs = [output.outputs[0].text for output in example_answers]
output_token_counts = []
for output in example_answers:
    candidate = output.outputs[0]
    if getattr(candidate, "token_ids", None) is not None:
        output_token_counts.append(len(candidate.token_ids))
    else:
        output_token_counts.append(
            len(tok.encode(candidate.text, add_special_tokens=False))
        )
avg_output_tokens = (
    sum(output_token_counts) / len(output_token_counts)
    if output_token_counts
    else 0.0
)

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
            "output_tokens": output_token_counts[i],
            "verified": bool(results[i]),
        }, ensure_ascii=False) + "\n")

total_time_seconds = time.perf_counter() - START_TIME

save_json(OUTPUT_FILE.replace(".jsonl", "_summary.json"), {
    "model": MODEL_PATH,
    "task": task,
    "steer_enable": STATIC_STEER_ENABLE,
    "steer_scale": STATIC_STEER_SCALE,
    "match_token_ids": STATIC_STEER_MATCH_TOKEN_IDS,
    "target_layer": STATIC_STEER_LAYER,
    "steer_vec_path": STATIC_STEER_PATH,
    "accuracy": accuracy,
    "avg_output_tokens": avg_output_tokens,
    "total_time_seconds": total_time_seconds,
    "num_samples": len(results),
    "results_file": OUTPUT_FILE,
})
print(accuracy)
print(f"avg_output_tokens: {avg_output_tokens:.2f}")
print(f"total_time_seconds: {total_time_seconds:.2f}")

