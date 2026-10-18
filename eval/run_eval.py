#!/usr/bin/env python3
"""
CI evaluation harness for PediJev prompt versions.

Runs every prompt file in prompts/ against the golden dataset
(data/golden_dataset.json + data/images/) using the SAME decision engine
as production (pipeline/pediev_pipeline.py: decide()), and reports
accuracy per prompt version.

Usage:
    OPENROUTER_API_KEY=... python eval/run_eval.py
    OPENROUTER_API_KEY=... python eval/run_eval.py --prompt prompts/pediev_v1_2.json
    OPENROUTER_API_KEY=... python eval/run_eval.py --fail-under 90

Writes:
    eval/results.json           -- full per-image results for every prompt tested
    $GITHUB_STEP_SUMMARY         -- markdown summary table (when running in Actions)
Exit code: 1 if any tested prompt's accuracy is below --fail-under (default: 0, i.e. report-only).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from pediev_pipeline import evaluate_image_file  # noqa: E402

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
GOLDEN_PATH = os.path.join(REPO_ROOT, "data", "golden_dataset.json")
IMAGES_DIR = os.path.join(REPO_ROOT, "data", "images")
PROMPTS_DIR = os.path.join(REPO_ROOT, "prompts")


def load_golden():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_prompt(prompt_path: str, golden: list) -> dict:
    correct, total = 0, 0
    per_image = []
    for item in golden:
        name = item["image_id"]
        gold = item.get("golden_decision")
        if not gold:
            continue
        img_path = os.path.join(IMAGES_DIR, f"{name}.jpg")
        if not os.path.exists(img_path):
            continue
        result = evaluate_image_file(img_path, prompt_path)
        decision = result.get("decision", "ERROR")
        agree = decision == gold
        total += 1
        correct += int(agree)
        per_image.append({
            "image_id": name,
            "golden": gold,
            "decision": decision,
            "agree": agree,
            "lat": result.get("lat"),
            "cost": result.get("cost"),
            "http_status": result.get("http_status"),
        })
    accuracy = (correct / total * 100) if total else 0.0
    return {
        "prompt_path": os.path.relpath(prompt_path, REPO_ROOT),
        "correct": correct,
        "total": total,
        "accuracy_pct": round(accuracy, 1),
        "per_image": per_image,
    }


def write_github_summary(all_results: list):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    lines = ["## PediJev prompt evaluation\n", "| Prompt | Accuracy | Mismatches |", "|---|---|---|"]
    for r in all_results:
        mismatches = [p["image_id"] for p in r["per_image"] if not p["agree"]]
        mismatch_str = ", ".join(mismatches) if mismatches else "-"
        lines.append(f"| `{r['prompt_path']}` | {r['correct']}/{r['total']} ({r['accuracy_pct']}%) | {mismatch_str} |")
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PediJev prompt versions against the golden dataset.")
    parser.add_argument(
        "--prompt", action="append",
        help="Path to a specific prompt JSON to evaluate. Repeatable. Default: evaluate every file in prompts/.",
    )
    parser.add_argument(
        "--fail-under", type=float, default=0.0,
        help="Exit non-zero if any evaluated prompt's accuracy is below this percent. Default: 0 (report-only, never fails).",
    )
    args = parser.parse_args()

    if "OPENROUTER_API_KEY" not in os.environ:
        print("ERROR: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(2)

    prompt_paths = args.prompt or sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.json")))
    if not prompt_paths:
        print("No prompt files found.", file=sys.stderr)
        sys.exit(2)

    golden = load_golden()
    all_results = []
    for p in prompt_paths:
        print(f"\n=== Evaluating {p} ===")
        result = evaluate_prompt(p, golden)
        for img in result["per_image"]:
            print(f"  {img['image_id']}: golden={img['golden']:6s} decision={img['decision']:10s} agree={img['agree']}")
        print(f"  -> {result['correct']}/{result['total']} ({result['accuracy_pct']}%)")
        all_results.append(result)

    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    write_github_summary(all_results)

    failing = [r for r in all_results if r["accuracy_pct"] < args.fail_under]
    if failing:
        print(f"\nFAIL: {len(failing)} prompt(s) below {args.fail_under}% threshold:", file=sys.stderr)
        for r in failing:
            print(f"  {r['prompt_path']}: {r['accuracy_pct']}%", file=sys.stderr)
        sys.exit(1)

    print("\nOK: all evaluated prompts meet the threshold.")


if __name__ == "__main__":
    main()
