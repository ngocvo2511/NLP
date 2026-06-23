from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .crf import LinearChainCRF
from .crf_data import (
    build_char_vocab,
    build_vocab,
    collate_bilstm,
    make_instances,
    spans_from_tag_ids,
    token_batches,
)
from .data import read_jsonl, write_jsonl
from .metrics import evaluate_exact
from .tags import build_sequence_labels


class CharCnnEncoder(nn.Module):
    def __init__(self, num_chars: int, char_dim: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_chars, char_dim, padding_idx=0)
        per_kernel = max(1, out_channels // 3)
        self.convs = nn.ModuleList(
            nn.Conv1d(char_dim, per_kernel, kernel_size=width, padding=0) for width in (2, 3, 4)
        )
        self.output_dim = per_kernel * len(self.convs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, word_len = char_ids.shape
        flat = char_ids.view(batch_size * seq_len, word_len)
        embedded = self.dropout(self.embedding(flat)).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            x = torch.relu(conv(embedded))
            pooled.append(torch.max(x, dim=2).values)
        return torch.cat(pooled, dim=1).view(batch_size, seq_len, -1)


class BiLstmCrf(nn.Module):
    def __init__(
        self,
        num_words: int,
        num_chars: int,
        num_tags: int,
        word_dim: int,
        char_dim: int,
        char_out: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.char_encoder = CharCnnEncoder(num_chars, char_dim, char_out, dropout)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            word_dim + self.char_encoder.output_dim,
            hidden_size // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(hidden_size, num_tags)
        self.crf = LinearChainCRF(num_tags)

    def emissions(self, token_ids: torch.Tensor, char_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        word_repr = self.word_embedding(token_ids)
        char_repr = self.char_encoder(char_ids)
        x = self.input_dropout(torch.cat([word_repr, char_repr], dim=-1))
        lengths = mask.long().sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=token_ids.size(1))
        return self.emission(self.output_dropout(encoded))

    def loss(self, token_ids: torch.Tensor, char_ids: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.crf.neg_log_likelihood(self.emissions(token_ids, char_ids, mask), tags, mask)

    def decode(self, token_ids: torch.Tensor, char_ids: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        return self.crf.decode(self.emissions(token_ids, char_ids, mask), mask)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(model, instances, examples, word_vocab, char_vocab, id2label, args, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in token_batches(instances, args.batch_tokens, shuffle=False):
            token_ids, char_ids, tags, mask = collate_bilstm(
                batch, word_vocab, char_vocab, lowercase=not args.no_lowercase, max_word_len=args.max_word_len
            )
            predictions = model.decode(token_ids.to(device), char_ids.to(device), mask.to(device))
            for inst, pred in zip(batch, predictions):
                spans = spans_from_tag_ids(pred, id2label, inst.offsets)
                rows.append({"text": inst.text, "labels": [[s.start, s.end, s.label] for s in spans]})
    pred_examples = read_temp_examples(rows)
    return evaluate_exact(examples, pred_examples), rows


def read_temp_examples(rows):
    from .data import Example, SpanLabel

    return [Example(row["text"], [SpanLabel(a, b, y) for a, b, y in row["labels"]]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a token/character BiLSTM-CRF ABSA model.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/bilstm-crf")
    parser.add_argument("--tag-scheme", default="bio", choices=["bio", "bilou"])
    parser.add_argument("--word-dim", type=int, default=300)
    parser.add_argument("--char-dim", type=int, default=64)
    parser.add_argument("--char-out", type=int, default=150)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch-tokens", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--min-word-freq", type=int, default=1)
    parser.add_argument("--max-word-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-lowercase", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = read_jsonl(Path(args.data_dir) / "train.jsonl")
    dev_examples = read_jsonl(Path(args.data_dir) / "dev.jsonl")
    span_labels = sorted({label.label for ex in train_examples + dev_examples for label in ex.labels})
    labels = build_sequence_labels(span_labels, args.tag_scheme)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    word_vocab = build_vocab(train_examples, min_freq=args.min_word_freq, lowercase=not args.no_lowercase)
    char_vocab = build_char_vocab(train_examples)

    train_instances = make_instances(train_examples, label2id, args.tag_scheme)
    dev_instances = make_instances(dev_examples, label2id, args.tag_scheme)

    model = BiLstmCrf(
        len(word_vocab),
        len(char_vocab),
        len(labels),
        args.word_dim,
        args.char_dim,
        args.char_out,
        args.hidden_size,
        args.layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = token_batches(train_instances, args.batch_tokens, shuffle=True)
        for batch in batches:
            token_ids, char_ids, tags, mask = collate_bilstm(
                batch, word_vocab, char_vocab, lowercase=not args.no_lowercase, max_word_len=args.max_word_len
            )
            loss = model.loss(token_ids.to(device), char_ids.to(device), tags.to(device), mask.to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        metrics, rows = evaluate_model(model, dev_instances, dev_examples, word_vocab, char_vocab, id2label, args, device)
        micro = metrics["micro"]
        record = {"epoch": epoch, "loss": total_loss / max(1, len(batches)), **micro}
        history.append(record)
        print(record)
        if micro["f1"] > best_f1:
            best_f1 = micro["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, output_dir / "best_model.pt")
            write_jsonl(output_dir / "dev_predictions.jsonl", rows)
            with (output_dir / "dev_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump({"word_vocab": word_vocab, "char_vocab": char_vocab, "labels": labels}, f, ensure_ascii=False, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
