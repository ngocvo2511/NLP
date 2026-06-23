from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .crf import LinearChainCRF
from .crf_data import build_char_vocab, collate_bilstm, make_instances, spans_from_tag_ids, token_batches
from .data import Example, SpanLabel, read_jsonl, write_jsonl
from .metrics import evaluate_exact
from .tags import build_sequence_labels
from .train_bilstm_crf import CharCnnEncoder


@dataclass
class TransformerCrfInstance:
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int, str]]
    tag_ids: list[int]
    input_ids: list[int]
    attention_mask: list[int]
    piece_to_word: list[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transformer_instances(examples, label2id, scheme, tokenizer, max_length):
    base_instances = make_instances(examples, label2id, scheme)
    return [encode_base_instance(base, tokenizer, max_length) for base in base_instances]


def encode_base_instance(base, tokenizer, max_length):
    max_content_length = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    pieces: list[str] = []
    piece_to_word_content: list[int] = []
    for word_idx, token in enumerate(base.tokens):
        subtokens = tokenizer.tokenize(token)
        if not subtokens:
            subtokens = [tokenizer.unk_token]
        for subtoken in subtokens:
            if len(pieces) >= max_content_length:
                break
            pieces.append(subtoken)
            piece_to_word_content.append(word_idx)
        if len(pieces) >= max_content_length:
            break

    piece_ids = tokenizer.convert_tokens_to_ids(pieces)
    special_mask = tokenizer.get_special_tokens_mask(piece_ids, already_has_special_tokens=False)
    input_ids = tokenizer.build_inputs_with_special_tokens(piece_ids)
    piece_to_word = []
    content_idx = 0
    for is_special in special_mask:
        if is_special:
            piece_to_word.append(-1)
        else:
            piece_to_word.append(piece_to_word_content[content_idx])
            content_idx += 1
    kept_word_count = max(piece_to_word_content) + 1 if piece_to_word_content else 0
    return TransformerCrfInstance(
        base.text,
        base.tokens[:kept_word_count],
        base.offsets[:kept_word_count],
        base.tag_ids[:kept_word_count],
        input_ids,
        [1] * len(input_ids),
        piece_to_word,
    )


def transformer_batches(instances: list[TransformerCrfInstance], batch_size: int, shuffle: bool):
    order = torch.randperm(len(instances)).tolist() if shuffle else list(range(len(instances)))
    for start in range(0, len(order), batch_size):
        yield [instances[i] for i in order[start : start + batch_size]]


def collate_transformer_crf(batch, tokenizer, char_vocab, max_word_len: int):
    max_piece_len = max(len(inst.input_ids) for inst in batch)
    max_word_len_seq = max(len(inst.tokens) for inst in batch)
    input_ids = torch.full((len(batch), max_piece_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_piece_len, dtype=torch.long)
    piece_to_word = torch.full((len(batch), max_piece_len), -1, dtype=torch.long)
    tag_ids = torch.zeros(len(batch), max_word_len_seq, dtype=torch.long)
    word_mask = torch.zeros(len(batch), max_word_len_seq, dtype=torch.bool)
    char_ids = torch.zeros(len(batch), max_word_len_seq, max_word_len, dtype=torch.long)

    for row, inst in enumerate(batch):
        input_ids[row, : len(inst.input_ids)] = torch.tensor(inst.input_ids)
        attention_mask[row, : len(inst.attention_mask)] = torch.tensor(inst.attention_mask)
        piece_to_word[row, : len(inst.piece_to_word)] = torch.tensor(inst.piece_to_word)
        for col, token in enumerate(inst.tokens):
            tag_ids[row, col] = inst.tag_ids[col]
            word_mask[row, col] = True
            for char_pos, char in enumerate(token[:max_word_len]):
                char_ids[row, col, char_pos] = char_vocab.get(char, 1)
    return input_ids, attention_mask, piece_to_word, char_ids, tag_ids, word_mask


class TransformerBiLstmCrf(nn.Module):
    def __init__(
        self,
        transformer,
        num_chars: int,
        num_tags: int,
        char_dim: int,
        char_out: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        freeze_transformer: bool,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        transformer_dim = transformer.config.hidden_size
        self.char_encoder = CharCnnEncoder(num_chars, char_dim, char_out, dropout)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            transformer_dim + self.char_encoder.output_dim,
            hidden_size // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.emission = nn.Linear(hidden_size, num_tags)
        self.crf = LinearChainCRF(num_tags)

    def word_representations(self, input_ids, attention_mask, piece_to_word, char_ids, word_mask):
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch_size, max_words = word_mask.shape
        reps = hidden.new_zeros(batch_size, max_words, hidden.size(-1))
        counts = hidden.new_zeros(batch_size, max_words, 1)
        for row in range(batch_size):
            valid = piece_to_word[row] >= 0
            if valid.any():
                word_ids = piece_to_word[row, valid]
                reps[row].index_add_(0, word_ids, hidden[row, valid])
                counts[row].index_add_(0, word_ids, torch.ones(len(word_ids), 1, device=hidden.device))
        reps = reps / counts.clamp_min(1.0)
        char_repr = self.char_encoder(char_ids)
        return self.input_dropout(torch.cat([reps, char_repr], dim=-1))

    def emissions(self, input_ids, attention_mask, piece_to_word, char_ids, word_mask):
        word_repr = self.word_representations(input_ids, attention_mask, piece_to_word, char_ids, word_mask)
        lengths = word_mask.long().sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(word_repr, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=word_mask.size(1))
        return self.emission(self.output_dropout(encoded))

    def loss(self, input_ids, attention_mask, piece_to_word, char_ids, tags, word_mask):
        emissions = self.emissions(input_ids, attention_mask, piece_to_word, char_ids, word_mask)
        return self.crf.neg_log_likelihood(emissions, tags, word_mask)

    def decode(self, input_ids, attention_mask, piece_to_word, char_ids, word_mask):
        emissions = self.emissions(input_ids, attention_mask, piece_to_word, char_ids, word_mask)
        return self.crf.decode(emissions, word_mask)


def evaluate_model(model, instances, examples, tokenizer, char_vocab, id2label, args, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in transformer_batches(instances, args.eval_batch_size, shuffle=False):
            input_ids, attention_mask, piece_to_word, char_ids, tags, word_mask = collate_transformer_crf(
                batch, tokenizer, char_vocab, args.max_word_len
            )
            preds = model.decode(
                input_ids.to(device),
                attention_mask.to(device),
                piece_to_word.to(device),
                char_ids.to(device),
                word_mask.to(device),
            )
            for inst, pred in zip(batch, preds):
                spans = spans_from_tag_ids(pred, id2label, inst.offsets)
                rows.append({"text": inst.text, "labels": [[s.start, s.end, s.label] for s in spans]})
    pred_examples = [Example(row["text"], [SpanLabel(a, b, y) for a, b, y in row["labels"]]) for row in rows]
    return evaluate_exact(examples, pred_examples), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer + BiLSTM-CRF for ABSA.")
    parser.add_argument("--model-name", default="vinai/phobert-base-v2")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/phobert-bilstm-crf")
    parser.add_argument("--tag-scheme", default="bio", choices=["bio", "bilou"])
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--char-dim", type=int, default=64)
    parser.add_argument("--char-out", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-word-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--freeze-transformer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

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
    char_vocab = build_char_vocab(train_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    transformer = AutoModel.from_pretrained(args.model_name)
    train_instances = make_transformer_instances(train_examples, label2id, args.tag_scheme, tokenizer, args.max_length)
    dev_instances = make_transformer_instances(dev_examples, label2id, args.tag_scheme, tokenizer, args.max_length)
    model = TransformerBiLstmCrf(
        transformer,
        len(char_vocab),
        len(labels),
        args.char_dim,
        args.char_out,
        args.hidden_size,
        args.layers,
        args.dropout,
        args.freeze_transformer,
    ).to(device)

    transformer_params = [p for n, p in model.named_parameters() if n.startswith("transformer.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("transformer.") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": transformer_params, "lr": args.learning_rate},
            {"params": head_params, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = list(transformer_batches(train_instances, args.batch_size, shuffle=True))
        for batch in batches:
            input_ids, attention_mask, piece_to_word, char_ids, tags, word_mask = collate_transformer_crf(
                batch, tokenizer, char_vocab, args.max_word_len
            )
            loss = model.loss(
                input_ids.to(device),
                attention_mask.to(device),
                piece_to_word.to(device),
                char_ids.to(device),
                tags.to(device),
                word_mask.to(device),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        metrics, rows = evaluate_model(model, dev_instances, dev_examples, tokenizer, char_vocab, id2label, args, device)
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

    tokenizer.save_pretrained(output_dir / "tokenizer")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": labels, "char_vocab": char_vocab, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
