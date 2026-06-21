from __future__ import annotations

import argparse
import json

from .data import read_jsonl
from .metrics import evaluate_exact


def print_table(title: str, rows: dict[str, dict[str, float]], limit: int | None = None) -> None:
    print(f"\n{title}")
    print(f"{'name':24} {'tp':>6} {'fp':>6} {'fn':>6} {'p':>8} {'r':>8} {'f1':>8}")
    sorted_rows = sorted(rows.items(), key=lambda item: item[1]["f1"])
    if limit is not None:
        sorted_rows = sorted_rows[:limit]
    for name, scores in sorted_rows:
        print(
            f"{name:24} {scores['tp']:6.0f} {scores['fp']:6.0f} {scores['fn']:6.0f} "
            f"{scores['precision']:8.4f} {scores['recall']:8.4f} {scores['f1']:8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate span-level ABSA predictions.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    gold = read_jsonl(args.gold)
    pred = read_jsonl(args.pred)
    if len(gold) != len(pred):
        raise ValueError(f"Gold and prediction size mismatch: {len(gold)} != {len(pred)}")

    result = evaluate_exact(gold, pred)
    micro = result["micro"]
    print("Exact span + label micro scores")
    print(f"  TP={micro['tp']} FP={micro['fp']} FN={micro['fn']}")
    print(f"  Precision={micro['precision']:.4f} Recall={micro['recall']:.4f} F1={micro['f1']:.4f}")
    print_table("By aspect", result["by_aspect"])
    print_table("By polarity", result["by_polarity"])
    print_table("Worst labels", result["by_label"], limit=15)

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
