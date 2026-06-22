from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from .data import read_jsonl
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


def encode_examples(examples, tokenizer, label2id: dict[str, int], max_length: int, tag_scheme: str) -> dict:
    texts = [ex.text for ex in examples]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )

    all_labels = []
    for row_idx, ex in enumerate(examples):
        offsets = tokenized["offset_mapping"][row_idx]
        labels = [-100] * len(offsets)

        for span in [label.clamped(ex.text) for label in ex.labels]:
            if span.start >= span.end:
                continue
            covered = [
                i
                for i, (start, end) in enumerate(offsets)
                if not (start == 0 and end == 0) and start < span.end and end > span.start
            ]
            for pos, token_idx in enumerate(covered):
                prefix = tag_prefix(pos, len(covered), tag_scheme)
                labels[token_idx] = label2id[f"{prefix}-{span.label}"]

        for i, (start, end) in enumerate(offsets):
            if labels[i] == -100:
                labels[i] = -100 if start == 0 and end == 0 else label2id["O"]

        all_labels.append(labels)

    tokenized.pop("offset_mapping")
    tokenized["labels"] = all_labels
    return tokenized


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


def compute_token_metrics(eval_pred, o_label_id: int) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    tp = fp = fn = correct = total = 0
    for pred_row, label_row in zip(predictions, labels):
        for pred_id, gold_id in zip(pred_row, label_row):
            if gold_id == -100:
                continue
            total += 1
            if pred_id == gold_id:
                correct += 1
            if pred_id != o_label_id and gold_id != o_label_id and pred_id == gold_id:
                tp += 1
            elif pred_id != o_label_id and pred_id != gold_id:
                fp += 1
            elif gold_id != o_label_id and pred_id != gold_id:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "token_accuracy": correct / total if total else 0.0,
        "token_non_o_precision": precision,
        "token_non_o_recall": recall,
        "token_non_o_f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer token classifier for ABSA.")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/xlmr-absa")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag-scheme", default="bio", choices=["bio", "bilou"])
    parser.add_argument("--class-weight", default="none", choices=["none", "balanced", "sqrt-balanced"])
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit(f"{args.model_name} does not provide a fast tokenizer with offset mappings.")

    train_encodings = encode_examples(train_examples, tokenizer, label2id, args.max_length, args.tag_scheme)
    dev_encodings = encode_examples(dev_examples, tokenizer, label2id, args.max_length, args.tag_scheme)
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
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "token_non_o_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
        "seed": args.seed,
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
        "compute_metrics": lambda pred: compute_token_metrics(pred, label2id["O"]),
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
