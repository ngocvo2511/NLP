from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch

from .data import Example, iter_token_spans
from .tags import spans_to_tags, spans_to_tags_with_offsets, tags_to_spans


PAD = "<pad>"
UNK = "<unk>"


@dataclass
class SequenceInstance:
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int, str]]
    tag_ids: list[int]


def iter_unit_spans(text: str, unit: str = "token") -> list[tuple[int, int, str]]:
    if unit == "token":
        return iter_token_spans(text)
    if unit == "char":
        return [(idx, idx + 1, char) for idx, char in enumerate(text)]
    raise ValueError(f"Unsupported sequence unit: {unit}")


def build_vocab(
    examples: Iterable[Example],
    min_freq: int = 1,
    lowercase: bool = True,
    unit: str = "token",
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ex in examples:
        for _, _, token in iter_unit_spans(ex.text, unit=unit):
            counts[token.lower() if lowercase else token] += 1
    vocab = {PAD: 0, UNK: 1}
    for token, count in counts.most_common():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_char_vocab(examples: Iterable[Example], min_freq: int = 1, unit: str = "token") -> dict[str, int]:
    counts: Counter[str] = Counter()
    for ex in examples:
        for _, _, token in iter_unit_spans(ex.text, unit=unit):
            counts.update(token)
    vocab = {PAD: 0, UNK: 1}
    for char, count in counts.most_common():
        if count >= min_freq and char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def make_instances(
    examples: Iterable[Example],
    label2id: dict[str, int],
    scheme: str,
    unit: str = "token",
) -> list[SequenceInstance]:
    instances = []
    for ex in examples:
        if unit == "token":
            tags, offsets = spans_to_tags(ex, scheme=scheme)
        else:
            offsets = iter_unit_spans(ex.text, unit=unit)
            tags, offsets = spans_to_tags_with_offsets(ex, offsets, scheme=scheme)
        tokens = [token for _, _, token in offsets]
        instances.append(SequenceInstance(ex.text, tokens, offsets, [label2id[tag] for tag in tags]))
    return instances


def token_batches(instances: list[SequenceInstance], batch_tokens: int, shuffle: bool = True) -> list[list[SequenceInstance]]:
    order = torch.randperm(len(instances)).tolist() if shuffle else list(range(len(instances)))
    ordered = [instances[i] for i in order]
    batches: list[list[SequenceInstance]] = []
    current: list[SequenceInstance] = []
    max_len = 0
    for inst in ordered:
        inst_len = max(1, len(inst.tokens))
        proposed_max = max(max_len, inst_len)
        if current and proposed_max * (len(current) + 1) > batch_tokens:
            batches.append(current)
            current = []
            max_len = 0
        current.append(inst)
        max_len = max(max_len, inst_len)
    if current:
        batches.append(current)
    return batches


def collate_bilstm(
    batch: list[SequenceInstance],
    word_vocab: dict[str, int],
    char_vocab: dict[str, int],
    lowercase: bool = True,
    max_word_len: int = 32,
):
    batch_size = len(batch)
    seq_len = max(len(inst.tokens) for inst in batch)
    token_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tag_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    char_ids = torch.zeros(batch_size, seq_len, max_word_len, dtype=torch.long)

    for row, inst in enumerate(batch):
        for col, token in enumerate(inst.tokens):
            key = token.lower() if lowercase else token
            token_ids[row, col] = word_vocab.get(key, word_vocab[UNK])
            tag_ids[row, col] = inst.tag_ids[col]
            mask[row, col] = True
            for char_pos, char in enumerate(token[:max_word_len]):
                char_ids[row, col, char_pos] = char_vocab.get(char, char_vocab[UNK])
    return token_ids, char_ids, tag_ids, mask


def spans_from_tag_ids(tag_ids: list[int], id2label: dict[int, str], offsets: list[tuple[int, int, str]]):
    return tags_to_spans([id2label[tag_id] for tag_id in tag_ids], offsets)
