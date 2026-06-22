from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import SpanLabel, read_jsonl, write_jsonl
from .postprocess import postprocess_spans


def require_deps():
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing transformer dependencies. Install with: pip install -r requirements.txt"
        ) from exc
    return torch, AutoModelForTokenClassification, AutoTokenizer


def tags_to_spans(tags: list[str], offsets: list[tuple[int, int]]) -> list[SpanLabel]:
    spans: list[SpanLabel] = []
    current_label = None
    current_start = None
    current_end = None

    def close_current() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            spans.append(SpanLabel(current_start, current_end, current_label))
        current_label = None
        current_start = None
        current_end = None

    for tag, (start, end) in zip(tags, offsets):
        if start == 0 and end == 0:
            continue
        if tag == "O" or "-" not in tag:
            close_current()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "U":
            close_current()
            spans.append(SpanLabel(start, end, label))
        elif prefix == "B" or label != current_label:
            close_current()
            current_label = label
            current_start = start
            current_end = end
            if prefix == "L":
                close_current()
        elif prefix == "L":
            current_end = end
            close_current()
        else:
            current_end = end
    close_current()
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict span-level ABSA labels with a fine-tuned transformer.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-span-chars", type=int, default=1)
    parser.add_argument("--merge-gap", type=int, default=-1)
    args = parser.parse_args()

    torch, AutoModelForTokenClassification, AutoTokenizer = require_deps()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    examples = read_jsonl(args.input)
    rows = []
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        texts = [ex.text for ex in batch]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").cpu().numpy().tolist()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.cpu().numpy()
        pred_ids = np.argmax(logits, axis=-1)

        for ex, row_ids, row_offsets in zip(batch, pred_ids, offsets):
            tags = [model.config.id2label[int(idx)] for idx in row_ids]
            spans = tags_to_spans(tags, [tuple(offset) for offset in row_offsets])
            spans = postprocess_spans(spans, ex.text, min_chars=args.min_span_chars, merge_gap=args.merge_gap)
            labels = [[span.start, span.end, span.label] for span in spans if span.start < span.end <= len(ex.text)]
            rows.append({"text": ex.text, "labels": labels})

    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
