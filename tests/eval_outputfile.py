import argparse
import json
import os
import re
import sys
from typing import Dict, List

from datasets import load_dataset

lcb_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "TensorRouter",
        "LiveCodeBench",
    )
)
if lcb_root not in sys.path:
    sys.path.append(lcb_root)

print(lcb_root)

def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] skip invalid json at line {line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def extract_python_code_block(text: str) -> str:
    matches = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return re.sub(r"</?think>", "", text).strip()


def load_lcb_eval_samples() -> Dict[str, dict]:
    # dataset = load_dataset(
    #     "livecodebench/code_generation_lite",
    #     version_tag="release_v6",
    #     split="test",
    # )
    dataset = load_dataset(
        'json',
        data_files="/home/jiayi/TensorRouter/TensorRouter/data/test6.jsonl",
        split="train",
    )

    lcb_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "LiveCodeBench"))
    if lcb_root not in sys.path:
        sys.path.append(lcb_root)

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    sample_map: Dict[str, dict] = {}
    for item in dataset:
        qid = str(item["question_id"])
        sample_map[qid] = CodeGenerationProblem(**dict(item)).get_evaluation_sample()

    print(len(sample_map))
    return sample_map


def evaluate_lcb_from_output(output_file: str) -> dict:
    rows = load_jsonl(output_file)
    if not rows:
        raise ValueError(f"No valid rows found in: {output_file}")

    sample_map = load_lcb_eval_samples()

    eval_samples: List[dict] = []
    generations: List[List[str]] = []

    for idx, row in enumerate(rows):
        qid = row.get("question_id")
        qid = str(qid)

        if qid not in sample_map:
            print(f"[warn] question_id {qid} not found in LCB test split; skipped")
            continue

        if isinstance(row.get("extracted_code"), str) and row["extracted_code"].strip():
            code = row["extracted_code"].strip()
        else:
            model_output = str(row.get("model_output", ""))
            code = extract_python_code_block(model_output)

        eval_samples.append(sample_map[qid])
        generations.append([code])

    from lcb_runner.evaluation import codegen_metrics

    print(eval_samples[0])
    print(generations[0])
    metrics, raw_results, raw_metadata = codegen_metrics(
        eval_samples,
        generations,
        k_list=[1],
        num_process_evaluate=int(os.getenv("LCB_NUM_PROCESS_EVALUATE", "16")),
        timeout=int(os.getenv("LCB_EVAL_TIMEOUT", "6")),
        debug=False,
    )

    sorted_ids = sorted(raw_results.keys())
    per_sample_pass = [all(v > 0 for v in raw_results[sid][0]) for sid in sorted_ids]

    return {
        "task": "livecodebench_v6",
        "output_file": output_file,
        "num_predictions": len(rows),
        "num_evaluated": len(eval_samples),
        "pass@1": float(metrics.get("pass@1", 0.0)),
        "accuracy": float(metrics.get("pass@1", 0.0)),
        "num_passed": int(sum(per_sample_pass)),
        "details": {
            "metrics": metrics,
            "raw_metadata_count": len(raw_metadata),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation-only script: load predictions from output JSONL and evaluate."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to prediction JSONL file. Expected fields include question_id and model_output/extracted_code.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save summary JSON.",
    )
    args = parser.parse_args()

    summary = evaluate_lcb_from_output(args.output_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary to: {args.save_json}")


if __name__ == "__main__":
    main()
