from __future__ import annotations

import argparse

from .data import SpanLabel, read_jsonl, write_jsonl
from .postprocess import postprocess_spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process span-level ABSA predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-span-chars", type=int, default=6)
    parser.add_argument("--merge-gap", type=int, default=30)
    args = parser.parse_args()

    examples = read_jsonl(args.input)
    rows = []
    for ex in examples:
        spans = [SpanLabel(span.start, span.end, span.label) for span in ex.labels]
        spans = postprocess_spans(spans, ex.text, min_chars=args.min_span_chars, merge_gap=args.merge_gap)
        rows.append({"text": ex.text, "labels": [[span.start, span.end, span.label] for span in spans]})

    write_jsonl(args.output, rows)
    print(f"Wrote post-processed predictions to {args.output}")


if __name__ == "__main__":
    main()
