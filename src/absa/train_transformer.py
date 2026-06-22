from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from .data import iter_token_spans, read_jsonl
from .metrics import prf
from .tags import build_sequence_labels, tag_prefix


def require_transformer_deps():
    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing transformer dependencies. Install with: pip install -r requirements.txt"
        ) from exc
    return torch, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed


class AbsDataset:
    def __init__(self, encodings: dict):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {key: value[idx] for key, value in self.encodings.items()}


def resolve_alignment_mode(tokenizer, requested: str) -> str:
    if requested == "auto":
        return "offset" if getattr(tokenizer, "is_fast", False) else "word"
    if requested == "offset" and not getattr(tokenizer, "is_fast", False):
        raise SystemExit(f"{tokenizer.name_or_path} does not provide a fast tokenizer with offset mappings.")
    return requested


def label_for_subtokens(
    text: str,
    token_offsets: list[tuple[int, int]],
    spans,
    label2id: dict[str, int],
    tag_scheme: str,
) -> list[int]:
    labels = [-100] * len(token_offsets)
    for span in [label.clamped(text) for label in spans]:
        if span.start >= span.end:
            continue
        covered = [
            i
            for i, (start, end) in enumerate(token_offsets)
            if not (start == 0 and end == 0) and start < span.end and end > span.start
        ]
        for pos, token_idx in enumerate(covered):
            prefix = tag_prefix(pos, len(covered), tag_scheme)
            labels[token_idx] = label2id[f"{prefix}-{span.label}"]
    return labels


def encode_with_offsets(examples, tokenizer, label2id: dict[str, int], max_length: int, tag_scheme: str) -> tuple[dict, list[list[tuple[int, int]]]]:
    texts = [ex.text for ex in examples]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )

    all_labels = []
    all_offsets = []
    for row_idx, ex in enumerate(examples):
        offsets = [tuple(offset) for offset in tokenized["offset_mapping"][row_idx]]
        labels = label_for_subtokens(ex.text, offsets, ex.labels, label2id, tag_scheme)

        for i, (start, end) in enumerate(offsets):
            if labels[i] == -100:
                labels[i] = -100 if start == 0 and end == 0 else label2id["O"]

        all_labels.append(labels)
        all_offsets.append(offsets)

    tokenized.pop("offset_mapping")
    tokenized["labels"] = all_labels
    return tokenized, all_offsets


def encode_with_word_alignment(examples, tokenizer, label2id: dict[str, int], max_length: int, tag_scheme: str) -> tuple[dict, list[list[tuple[int, int]]]]:
    input_ids = []
    attention_mask = []
    all_labels = []
    all_offsets = []
    max_content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)

    for ex in examples:
        pieces = []
        piece_offsets = []
        for start, end, token in iter_token_spans(ex.text):
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

        row_labels = label_for_subtokens(ex.text, row_offsets, ex.labels, label2id, tag_scheme)
        for i, (start, end) in enumerate(row_offsets):
            if row_labels[i] == -100:
                row_labels[i] = -100 if start == 0 and end == 0 else label2id["O"]

        input_ids.append(row_input_ids)
        attention_mask.append([1] * len(row_input_ids))
        all_labels.append(row_labels)
        all_offsets.append(row_offsets)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": all_labels}, all_offsets


def encode_examples(
    examples,
    tokenizer,
    label2id: dict[str, int],
    max_length: int,
    tag_scheme: str,
    alignment_mode: str,
) -> tuple[dict, list[list[tuple[int, int]]]]:
    if alignment_mode == "offset":
        return encode_with_offsets(examples, tokenizer, label2id, max_length, tag_scheme)
    if alignment_mode == "word":
        return encode_with_word_alignment(examples, tokenizer, label2id, max_length, tag_scheme)
    raise ValueError(f"Unsupported tokenizer alignment mode: {alignment_mode}")


def make_class_weights(label_rows: list[list[int]], num_labels: int, mode: str, torch):
    if mode == "none":
        return None

    counts = np.zeros(num_labels, dtype=np.float64)
    for row in label_rows:
        for label_id in row:
            if label_id != -100:
                counts[label_id] += 1

    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return None

    total = nonzero.sum()
    weights = np.ones(num_labels, dtype=np.float32)
    for idx, count in enumerate(counts):
        if count > 0:
            balanced = total / (len(nonzero) * count)
            weights[idx] = balanced ** 0.5 if mode == "sqrt-balanced" else balanced
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_trainer(base_trainer, class_weights, torch):
    class WeightedTokenClassificationTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            weights = class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTokenClassificationTrainer


def ids_to_spans(pred_ids: np.ndarray, offsets: list[tuple[int, int]], id2label: dict[int, str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def close_current() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            spans.add((current_start, current_end, current_label))
        current_label = None
        current_start = None
        current_end = None

    for pred_id, (start, end) in zip(pred_ids[: len(offsets)], offsets):
        if start == 0 and end == 0:
            continue
        tag = id2label[int(pred_id)]
        if tag == "O" or "-" not in tag:
            close_current()
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "U":
            close_current()
            spans.add((start, end, label))
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


def make_compute_metrics(dev_examples, dev_offsets, id2label: dict[int, str], o_label_id: int):
    gold_sets = []
    for ex in dev_examples:
        gold_sets.append(
            {
                (span.clamped(ex.text).start, span.clamped(ex.text).end, span.label)
                for span in ex.labels
                if span.clamped(ex.text).start < span.clamped(ex.text).end
            }
        )

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        token_tp = token_fp = token_fn = correct = total = 0
        span_tp = span_fp = span_fn = 0

        for row_idx, (pred_row, label_row) in enumerate(zip(predictions, labels)):
            for pred_id, gold_id in zip(pred_row, label_row):
                if gold_id == -100:
                    continue
                total += 1
                if pred_id == gold_id:
                    correct += 1
                if pred_id != o_label_id and gold_id != o_label_id and pred_id == gold_id:
                    token_tp += 1
                elif pred_id != o_label_id and pred_id != gold_id:
                    token_fp += 1
                elif gold_id != o_label_id and pred_id != gold_id:
                    token_fn += 1

            pred_spans = ids_to_spans(pred_row, dev_offsets[row_idx], id2label)
            gold_spans = gold_sets[row_idx]
            span_tp += len(pred_spans & gold_spans)
            span_fp += len(pred_spans - gold_spans)
            span_fn += len(gold_spans - pred_spans)

        token_scores = prf(token_tp, token_fp, token_fn)
        span_scores = prf(span_tp, span_fp, span_fn)
        return {
            "token_accuracy": correct / total if total else 0.0,
            "token_non_o_precision": token_scores["precision"],
            "token_non_o_recall": token_scores["recall"],
            "token_non_o_f1": token_scores["f1"],
            "exact_span_precision": span_scores["precision"],
            "exact_span_recall": span_scores["recall"],
            "exact_span_f1": span_scores["f1"],
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer token classifier for ABSA.")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/xlmr-absa")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag-scheme", default="bio", choices=["bio", "bilou"])
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced", "sqrt-balanced"])
    parser.add_argument("--metric-for-best-model", default="exact_span_f1")
    parser.add_argument("--tokenizer-alignment", default="auto", choices=["auto", "offset", "word"])
    args = parser.parse_args()

    (
        torch,
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    ) = require_transformer_deps()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_examples = read_jsonl(data_dir / "train.jsonl")
    dev_examples = read_jsonl(data_dir / "dev.jsonl")
    span_labels = sorted({label.label for ex in train_examples + dev_examples for label in ex.labels})
    labels = build_sequence_labels(span_labels, scheme=args.tag_scheme)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.tokenizer_alignment != "word")
    alignment_mode = resolve_alignment_mode(tokenizer, args.tokenizer_alignment)
    print(f"Using tokenizer alignment mode: {alignment_mode}")

    train_encodings, _ = encode_examples(train_examples, tokenizer, label2id, args.max_length, args.tag_scheme, alignment_mode)
    dev_encodings, dev_offsets = encode_examples(dev_examples, tokenizer, label2id, args.max_length, args.tag_scheme, alignment_mode)
    train_dataset = AbsDataset(train_encodings)
    dev_dataset = AbsDataset(dev_encodings)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
        "seed": args.seed,
        "fp16": args.fp16,
    }
    args_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in args_signature.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    class_weights = make_class_weights(train_encodings["labels"], len(labels), args.class_weight, torch)
    trainer_class = make_weighted_trainer(Trainer, class_weights, torch) if class_weights is not None else Trainer

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator,
        "compute_metrics": make_compute_metrics(dev_examples, dev_offsets, id2label, label2id["O"]),
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trainer_class(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "labels.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "label2id": label2id,
                "id2label": id2label,
                "tag_scheme": args.tag_scheme,
                "class_weight": args.class_weight,
                "tokenizer_alignment": alignment_mode,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(Path(args.output_dir) / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
