from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import SpanLabel, iter_token_spans, read_jsonl, write_jsonl
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


def resolve_alignment_mode(model_dir: str, tokenizer) -> str:
    labels_path = Path(model_dir) / "labels.json"
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labels_meta = json.load(f)
        mode = labels_meta.get("tokenizer_alignment")
        if mode:
            return mode
    return "offset" if getattr(tokenizer, "is_fast", False) else "word"


def encode_word_aligned_batch(texts: list[str], tokenizer, max_length: int):
    input_ids = []
    attention_mask = []
    offsets = []
    max_content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)

    for text in texts:
        pieces = []
        piece_offsets = []
        for start, end, token in iter_token_spans(text):
            subtokens = tokenizer.tokenize(token)
            if not subtokens:
                subtokens = [tokenizer.unk_token]
            pieces.extend(subtokens)
            piece_offsets.extend((start, end) for _ in subtokens)

        pieces = pieces[:max_content_length]
        piece_offsets = piece_offsets[:max_content_length]
        piece_ids = tokenizer.convert_tokens_to_ids(pieces)
        special_mask = tokenizer.get_special_tokens_mask(piece_ids, already_has_special_tokens=False)
        row_input_ids = tokenizer.build_inputs_with_special_tokens(piece_ids)

        row_offsets = []
        piece_idx = 0
        for is_special in special_mask:
            if is_special:
                row_offsets.append((0, 0))
            else:
                row_offsets.append(piece_offsets[piece_idx])
                piece_idx += 1

        input_ids.append(row_input_ids)
        attention_mask.append([1] * len(row_input_ids))
        offsets.append(row_offsets)

    return {"input_ids": input_ids, "attention_mask": attention_mask}, offsets


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
    alignment_mode = resolve_alignment_mode(args.model_dir, tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    examples = read_jsonl(args.input)
    rows = []
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        texts = [ex.text for ex in batch]
        if alignment_mode == "word":
            encoded, offsets = encode_word_aligned_batch(texts, tokenizer, args.max_length)
            encoded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        else:
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
