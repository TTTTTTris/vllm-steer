# tests/test_steer_serve.py
import json
import os
import re
import signal
import subprocess
import threading
import time
from typing import Optional, TextIO

import requests
from datasets import load_dataset
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
from transformers import AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# STEER_VEC_PATH = ""
# STEER_VEC_PATH = "/home/jiayi_tian/TensorRouter/TensorRouter/vector_500_500/DeepSeek-R1-Distill-Qwen-1.5B/layer_20_transition_reflection_steervec.pt"
STEER_VEC_PATH = "/home/jiayi_tian/TensorRouter/TensorRouter/vector_500_500/DeepSeek-R1-Distill-Qwen-1.5B/layer_21_highrank_60_transition_reflection_steervec.pt"
TARGET_LAYER = 20
STEER_SCALE = float(os.getenv("STEER_SCALE", "0"))
STATIC_STEER_ENABLE = os.getenv("STATIC_STEER_ENABLE", "0")
STATIC_STEER_DEBUG = os.getenv("STATIC_STEER_DEBUG", "0")
STATIC_STEER_DEBUG_EVERY = os.getenv("STATIC_STEER_DEBUG_EVERY", "1")
STATIC_STEER_DEBUG_MAX_PRINTS = os.getenv("STATIC_STEER_DEBUG_MAX_PRINTS", "500")


tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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

STATIC_STEER_MATCH_TOKEN_ID = tok.encode("\n\n", add_special_tokens=False)[0]

GPU = os.getenv("GPU", "5")

HOST = "localhost"
PORT = os.getenv("PORT", "8008")
BASE_URL = f"http://{HOST}:{PORT}"
OPENAI_API_KEY = "EMPTY"

NUM_SAMPLES = 1
MAX_TOKENS = 16384
task='aime_2024'

OUTPUT_FILE = f"results/{task}_serve_results_{STATIC_STEER_ENABLE}_{STEER_SCALE}_debug_{NUM_SAMPLES}.jsonl"
print(OUTPUT_FILE)
SUMMARY_FILE = OUTPUT_FILE.replace(".jsonl", "_summary.json")

import time


class ServerLogReader:
    def __init__(self, stream: TextIO, keep_lines: int = 4000, echo: bool = True):
        self._stream = stream
        self._keep_lines = keep_lines
        self._echo = echo
        self._lines = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        for line in iter(self._stream.readline, ""):
            text = line.rstrip("\n")
            if self._echo:
                print(text, flush=True)
            with self._lock:
                self._lines.append(text)
                if len(self._lines) > self._keep_lines:
                    self._lines = self._lines[-self._keep_lines :]

    def contains(self, needle: str) -> bool:
        with self._lock:
            return any(needle in line for line in self._lines)

    def wait_for(self, needle: str, timeout_s: int) -> bool:
        start = time.time()
        while time.time() - start < timeout_s:
            if self.contains(needle):
                return True
            time.sleep(0.2)
        return False

    def tail(self, n: int = 120) -> str:
        with self._lock:
            return "\n".join(self._lines[-n:])


def validate_static_steer_config():
    if STATIC_STEER_ENABLE != "1":
        return
    if not STEER_VEC_PATH:
        raise ValueError("STATIC_STEER_ENABLE=1 but STEER_VEC_PATH is empty")
    # if abs(float(STEER_SCALE)) < 1e-12:
        # raise ValueError("STATIC_STEER_ENABLE=1 but STEER_SCALE is 0, no steering effect")

def wait_until_server_ready(
    base_url: str,
    proc: subprocess.Popen,
    log_reader: ServerLogReader,
    timeout_s: int = 300,
):
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                "Server process exited before becoming ready. "
                f"exit_code={proc.returncode}\n"
                f"Server log tail:\n{log_reader.tail()}"
            )
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.ok:
                return
        except Exception as e:
            last_err = e
        time.sleep(2)
    raise RuntimeError(
        "Server did not become ready within "
        f"{timeout_s}s. Last error: {last_err}\n"
        f"Server log tail:\n{log_reader.tail()}"
    )


def generate_one(problem: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": build_messages(problem),
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=600,
    )
    if not r.ok:
        print("status:", r.status_code)
        print("response:", r.text)
        print("payload:", json.dumps(payload, indent=2)[:4000])
        r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]
    
def extract_box(pred_str: str) -> str:
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    if ans[0] == "{":
        stack = 1
        out = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                out += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                out += c
            else:
                out += c
        return out.strip()
    return ans.split("$")[0].strip()


def build_messages(problem: str):
    return [
        {
            "role": "user",
            "content": (
                "Please reason step by step, and put your final answer within \\boxed{}.\n"
                f"{problem}"
            ),
        }
    ]

def start_server():
    env = os.environ.copy()
    env["HF_HOME"] = "/raid0-data/jiayi_tian"
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env["VLLM_USE_PRECOMPILED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # Steering config consumed by the server-side startup hook.
    env["STATIC_STEER_ENABLE"] = STATIC_STEER_ENABLE
    env["STATIC_STEER_PATH"] = STEER_VEC_PATH
    env["STATIC_STEER_LAYER"] = str(TARGET_LAYER)
    env["STATIC_STEER_SCALE"] = str(STEER_SCALE)
    env["STATIC_STEER_MATCH_TOKEN_IDS"] = str(STATIC_STEER_MATCH_TOKEN_ID)
    env["STATIC_STEER_DEBUG"] = STATIC_STEER_DEBUG
    env["STATIC_STEER_DEBUG_EVERY"] = STATIC_STEER_DEBUG_EVERY
    env["STATIC_STEER_DEBUG_MAX_PRINTS"] = STATIC_STEER_DEBUG_MAX_PRINTS

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host", HOST,
        "--port", str(PORT),
        "--tensor-parallel-size", "1",
        "--api-key", OPENAI_API_KEY,
        "--max-model-len", f"{MAX_TOKENS+2000}",
        # "--gpu-memory-utilization", "0.5",
    ]

    # Popen requires command args and env values to be strings/path-like.
    cmd = [str(x) for x in cmd]
    env = {k: str(v) for k, v in env.items()}

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )

    log_reader = ServerLogReader(proc.stdout)
    log_reader.start()

    return proc, log_reader


def stop_server(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()

def main():
    run_start = time.time()
    validate_static_steer_config()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    data = load_dataset(f"HuggingFaceH4/{task}", split="train")
    if NUM_SAMPLES > 0:
        data = data.select(range(NUM_SAMPLES))

    problems, answers = [], []
    for ex in data:
        problems.append(ex["problem"])
        answers.append(ex["answer"])

    proc, log_reader = start_server()
    try:
        wait_until_server_ready(BASE_URL, proc, log_reader)

        if STATIC_STEER_ENABLE == "1":
            steer_markers = [
                "[static-steer] enabling steering",
                "[static-steer] non-eager mode detected",
            ]
            marker_found = any(
                log_reader.wait_for(marker, timeout_s=30)
                for marker in steer_markers
            )
            if not marker_found:
                raise RuntimeError(
                    "Expected static steering startup log not found. "
                    "Server log tail:\n"
                    f"{log_reader.tail()}"
                )
            print("[check] static steering startup hook confirmed by server log")

        outputs = []
        for i, problem in enumerate(problems):
            print(f"[{i+1}/{len(problems)}] decoding...")
            outputs.append(generate_one(problem))

        extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
        results = []
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for i, out in enumerate(outputs):
                gold = parse(f"${answers[i]}$", extraction_config=extraction_target)
                pred = parse(out, extraction_config=extraction_target)
                ok = verify(gold, pred)
                results.append(bool(ok))
                f.write(json.dumps({
                    "index": i,
                    "problem": problems[i],
                    "gold_answer": answers[i],
                    "boxed_pred": extract_box(out),
                    "model_output": out,
                    "verified": bool(ok),
                }, ensure_ascii=False) + "\n")

        acc = sum(results) / len(results)
        elapsed_s = time.time() - run_start
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL_NAME,
                "steer_vec_path": STEER_VEC_PATH,
                "target_layer": TARGET_LAYER,
                "steer_scale": STEER_SCALE,
                "num_samples": len(results),
                "accuracy": acc,
                "elapsed_seconds": elapsed_s,
                "elapsed_minutes": elapsed_s / 60.0,
                "results_file": OUTPUT_FILE,
            }, f, ensure_ascii=False, indent=2)

        print(f"accuracy = {acc:.4f}")
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()