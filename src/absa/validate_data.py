from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from .data import read_jsonl, split_aspect_polarity


def summarize_split(path: Path) -> None:
    examples = read_jsonl(path)
    labels = [label.label for ex in examples for label in ex.labels]
    aspects = Counter(split_aspect_polarity(label)[0] for label in labels)
    polarities = Counter(split_aspect_polarity(label)[1] for label in labels)
    text_lengths = [len(ex.text) for ex in examples]
    span_lengths = [max(0, min(label.end, len(ex.text)) - max(0, label.start)) for ex in examples for label in ex.labels]

    bad_offsets = 0
    empty_spans = 0
    overlap_rows = 0
    for ex in examples:
        spans = sorted(ex.labels, key=lambda x: (x.start, x.end, x.label))
        for label in spans:
            if not (0 <= label.start < label.end <= len(ex.text)):
                bad_offsets += 1
            clamped = label.clamped(ex.text)
            if not ex.text[clamped.start:clamped.end].strip():
                empty_spans += 1
        if any(next_label.start < label.end for label, next_label in zip(spans, spans[1:])):
            overlap_rows += 1

    print(f"\n{path.name}")
    print(f"  rows: {len(examples)}")
    print(f"  labels: {len(labels)}")
    print(f"  avg labels / row: {len(labels) / len(examples):.2f}")
    print(f"  text length avg/max: {mean(text_lengths):.1f}/{max(text_lengths)}")
    print(f"  span length avg/max: {mean(span_lengths):.1f}/{max(span_lengths)}")
    print(f"  bad offsets: {bad_offsets}")
    print(f"  empty spans after clamp: {empty_spans}")
    print(f"  rows with overlapping spans: {overlap_rows}")
    print(f"  aspects: {dict(aspects.most_common())}")
    print(f"  polarities: {dict(polarities.most_common())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ABSA JSONL dataset splits.")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for split in ("train", "dev", "test"):
        summarize_split(data_dir / f"{split}.jsonl")


if __name__ == "__main__":
    main()
