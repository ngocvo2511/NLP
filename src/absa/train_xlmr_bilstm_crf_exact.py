from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .crf import LinearChainCRF
from .data import Example, SpanLabel, iter_token_spans, read_jsonl, write_jsonl
from .metrics import evaluate_exact
from .tags import build_sequence_labels, spans_to_tags, spans_to_tags_with_offsets, tags_to_spans


PAD = "<PAD>"
UNK = "<UNK>"


class Vocab:
    def __init__(self) -> None:
        self.token2id = {PAD: 0, UNK: 1}

    def add(self, token: str) -> None:
        if token not in self.token2id:
            self.token2id[token] = len(self.token2id)

    def get_id(self, token: str) -> int:
        return self.token2id.get(token, self.token2id[UNK])

    def __len__(self) -> int:
        return len(self.token2id)

    def to_dict(self) -> dict[str, int]:
        return self.token2id


@dataclass
class Feature:
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int, str]]
    input_ids: list[int]
    attention_mask: list[int]
    first_subword_indices: list[int]
    word_ids: list[int]
    char_ids: list[list[int]]
    labels: list[int]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_sequence_spans(text: str, unit: str) -> list[tuple[int, int, str]]:
    if unit == "token":
        return iter_token_spans(text)
    if unit == "char":
        return [(idx, idx + 1, char) for idx, char in enumerate(text)]
    raise ValueError(f"Unsupported sequence unit: {unit}")


def build_vocabs(examples: list[Example], lowercase: bool = True, unit: str = "token") -> tuple[Vocab, Vocab]:
    word_vocab = Vocab()
    char_vocab = Vocab()
    for ex in examples:
        for _, _, token in iter_sequence_spans(ex.text, unit):
            word_vocab.add(token.lower() if lowercase else token)
            for char in token:
                char_vocab.add(char)
    return word_vocab, char_vocab


class ExactCrfDataset(Dataset):
    def __init__(self, examples, word_vocab, char_vocab, label2id, tokenizer, max_length, lowercase=True, unit="token"):
        self.examples = examples
        self.features = []
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lowercase = lowercase
        self.unit = unit
        self._build()

    def _build(self) -> None:
        for ex in self.examples:
            if self.unit == "token":
                tag_strings, offsets = spans_to_tags(ex, scheme="bio")
            else:
                offsets = iter_sequence_spans(ex.text, self.unit)
                tag_strings, offsets = spans_to_tags_with_offsets(ex, offsets, scheme="bio")
            tokens = [token for _, _, token in offsets]
            if self.unit == "token":
                encoded = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                )
                hf_word_ids = encoded.word_ids()
                first_subword_by_word: dict[int, int] = {}
                for subword_idx, word_idx in enumerate(hf_word_ids):
                    if word_idx is not None and word_idx not in first_subword_by_word:
                        first_subword_by_word[word_idx] = subword_idx
                kept_word_ids = sorted(first_subword_by_word)
                if not kept_word_ids:
                    continue
                kept_word_count = max(kept_word_ids) + 1
                first_subword_indices = [first_subword_by_word[i] for i in range(kept_word_count)]
            else:
                encoded = self.tokenizer(
                    ex.text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                    return_offsets_mapping=True,
                )
                first_subword_by_char: dict[int, int] = {}
                max_covered_char = 0
                for subword_idx, span in enumerate(encoded.pop("offset_mapping")):
                    start, end = span
                    if end <= start:
                        continue
                    max_covered_char = max(max_covered_char, end)
                    for char_idx in range(start, end):
                        first_subword_by_char.setdefault(char_idx, subword_idx)
                kept_word_count = 0
                first_subword_indices = []
                for char_idx in range(len(tokens)):
                    if char_idx >= len(ex.text):
                        break
                    if char_idx >= max_covered_char:
                        break
                    first_subword_indices.append(first_subword_by_char.get(char_idx, 0))
                    kept_word_count += 1
                if kept_word_count == 0:
                    continue
            kept_tokens = tokens[:kept_word_count]
            self.features.append(
                Feature(
                    text=ex.text,
                    tokens=kept_tokens,
                    offsets=offsets[:kept_word_count],
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    first_subword_indices=first_subword_indices,
                    word_ids=[self.word_vocab.get_id(token.lower() if self.lowercase else token) for token in kept_tokens],
                    char_ids=[[self.char_vocab.get_id(char) for char in token] for token in kept_tokens],
                    labels=[self.label2id[tag] for tag in tag_strings[:kept_word_count]],
                )
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_exact(batch: list[Feature], pad_token_id: int):
    batch_size = len(batch)
    max_subwords = max(len(x.input_ids) for x in batch)
    max_words = max(len(x.word_ids) for x in batch)
    max_word_len = max(max((len(chars) for chars in x.char_ids), default=1) for x in batch)

    input_ids = torch.full((batch_size, max_subwords), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_subwords, dtype=torch.long)
    word_ids = torch.zeros(batch_size, max_words, dtype=torch.long)
    char_ids = torch.zeros(batch_size, max_words, max_word_len, dtype=torch.long)
    word_lengths = torch.zeros(batch_size, max_words, dtype=torch.long)
    first_subword_indices = torch.zeros(batch_size, max_words, dtype=torch.long)
    labels = torch.zeros(batch_size, max_words, dtype=torch.long)
    word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)

    for row, feature in enumerate(batch):
        input_ids[row, : len(feature.input_ids)] = torch.tensor(feature.input_ids)
        attention_mask[row, : len(feature.attention_mask)] = torch.tensor(feature.attention_mask)
        for col, word_id in enumerate(feature.word_ids):
            word_ids[row, col] = word_id
            labels[row, col] = feature.labels[col]
            word_mask[row, col] = True
            first_subword_indices[row, col] = feature.first_subword_indices[col]
            word_lengths[row, col] = max(1, len(feature.char_ids[col]))
            char_ids[row, col, : len(feature.char_ids[col])] = torch.tensor(feature.char_ids[col])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_ids": word_ids,
        "char_ids": char_ids,
        "word_lengths": word_lengths,
        "first_subword_indices": first_subword_indices,
        "labels": labels,
        "word_mask": word_mask,
        "features": batch,
    }


class CharLSTM(nn.Module):
    def __init__(self, char_vocab_size: int, char_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(char_emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.output_dim = hidden_dim * 2

    def forward(self, char_ids: torch.Tensor, word_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_word_len = char_ids.shape
        flat_chars = char_ids.view(batch_size * seq_len, max_word_len)
        flat_lengths = word_lengths.view(-1).cpu().clamp_min(1)
        embeds = self.char_embedding(flat_chars)
        lengths, sort_idx = flat_lengths.sort(descending=True)
        packed = nn.utils.rnn.pack_padded_sequence(embeds[sort_idx], lengths, batch_first=True)
        _, (hidden, _) = self.lstm(packed)
        char_repr = torch.cat([hidden[0], hidden[1]], dim=-1)
        _, unsort_idx = sort_idx.sort()
        char_repr = char_repr[unsort_idx]
        return char_repr.view(batch_size, seq_len, -1)


class XlmrBiLstmCrfExact(nn.Module):
    def __init__(
        self,
        transformer,
        vocab_size: int,
        char_vocab_size: int,
        num_labels: int,
        syllable_dim: int,
        char_emb_dim: int,
        char_hidden_dim: int,
        lstm_hidden_size: int,
        dropout: float,
        freeze_transformer: bool,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        self.syllable_embedding = nn.Embedding(vocab_size, syllable_dim, padding_idx=0)
        self.char_lstm = CharLSTM(char_vocab_size, char_emb_dim, char_hidden_dim)
        total_dim = transformer.config.hidden_size + syllable_dim + self.char_lstm.output_dim
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(total_dim, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = LinearChainCRF(num_labels)

    def emissions(self, input_ids, attention_mask, word_ids, char_ids, word_lengths, first_subword_indices, word_mask):
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        gather_idx = first_subword_indices.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        contextual = hidden.gather(dim=1, index=gather_idx)
        syllable = self.syllable_embedding(word_ids)
        chars = self.char_lstm(char_ids, word_lengths)
        combined = self.dropout(torch.cat([contextual, syllable, chars], dim=-1))
        lengths = word_mask.long().sum(dim=1).cpu().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(combined, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.bilstm(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=word_ids.size(1))
        return self.hidden2tag(self.dropout(encoded))

    def loss(self, batch):
        emissions = self.emissions(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_ids"],
            batch["char_ids"],
            batch["word_lengths"],
            batch["first_subword_indices"],
            batch["word_mask"],
        )
        return self.crf.neg_log_likelihood(emissions, batch["labels"], batch["word_mask"])

    def decode(self, batch):
        emissions = self.emissions(
            batch["input_ids"],
            batch["attention_mask"],
            batch["word_ids"],
            batch["char_ids"],
            batch["word_lengths"],
            batch["first_subword_indices"],
            batch["word_mask"],
        )
        return self.crf.decode(emissions, batch["word_mask"])


def move_batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate_model(model, loader, gold_examples, id2label, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"]
            batch = move_batch_to_device(batch, device)
            paths = model.decode(batch)
            for feature, path in zip(features, paths):
                tags = [id2label[tag_id] for tag_id in path]
                spans = tags_to_spans(tags, feature.offsets)
                rows.append({"text": feature.text, "labels": [[span.start, span.end, span.label] for span in spans]})
    pred_examples = [Example(row["text"], [SpanLabel(a, b, y) for a, b, y in row["labels"]]) for row in rows]
    return evaluate_exact(gold_examples, pred_examples), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XLM-R + syllable/char BiLSTM-CRF with exact span evaluation.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/xlmr-bilstm-crf-exact")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--unit", default="token", choices=["token", "char"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--transformer-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--syllable-dim", type=int, default=100)
    parser.add_argument("--char-emb-dim", type=int, default=50)
    parser.add_argument("--char-hidden-dim", type=int, default=50)
    parser.add_argument("--lstm-hidden-size", type=int, default=400)
    parser.add_argument("--dropout", type=float, default=0.33)
    parser.add_argument("--freeze-transformer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = read_jsonl(Path(args.data_dir) / "train.jsonl")
    dev_examples = read_jsonl(Path(args.data_dir) / "dev.jsonl")
    test_examples = read_jsonl(Path(args.data_dir) / "test.jsonl") if args.eval_test else []
    word_vocab, char_vocab = build_vocabs(train_examples, unit=args.unit)
    span_labels = sorted({label.label for ex in train_examples for label in ex.labels})
    labels = build_sequence_labels(span_labels, scheme="bio")
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_dataset = ExactCrfDataset(
        train_examples, word_vocab, char_vocab, label2id, tokenizer, args.max_length, unit=args.unit
    )
    dev_dataset = ExactCrfDataset(
        dev_examples, word_vocab, char_vocab, label2id, tokenizer, args.max_length, unit=args.unit
    )
    test_dataset = (
        ExactCrfDataset(test_examples, word_vocab, char_vocab, label2id, tokenizer, args.max_length, unit=args.unit)
        if args.eval_test
        else None
    )
    collate = lambda batch: collate_exact(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate) if test_dataset else None

    model = XlmrBiLstmCrfExact(
        AutoModel.from_pretrained(args.model_name),
        len(word_vocab),
        len(char_vocab),
        len(labels),
        args.syllable_dim,
        args.char_emb_dim,
        args.char_hidden_dim,
        args.lstm_hidden_size,
        args.dropout,
        args.freeze_transformer,
    ).to(device)
    transformer_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("transformer."):
            transformer_params.append(param)
        else:
            head_params.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": transformer_params, "lr": args.transformer_lr}, {"params": head_params, "lr": args.head_lr}]
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        dev_metrics, dev_rows = evaluate_model(model, dev_loader, dev_examples, id2label, device)
        micro = dev_metrics["micro"]
        record = {"epoch": epoch, "loss": total_loss / max(1, len(train_loader)), **micro}
        history.append(record)
        print(record)
        if micro["f1"] > best_f1:
            best_f1 = micro["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, output_dir / "best_model.pt")
            write_jsonl(output_dir / "dev_predictions.jsonl", dev_rows)
            with (output_dir / "dev_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, ensure_ascii=False, indent=2)

    if test_loader is not None:
        checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        test_metrics, test_rows = evaluate_model(model, test_loader, test_examples, id2label, device)
        write_jsonl(output_dir / "test_predictions.jsonl", test_rows)
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "word_vocab": word_vocab.to_dict(),
                "char_vocab": char_vocab.to_dict(),
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
