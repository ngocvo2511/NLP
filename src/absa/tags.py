from __future__ import annotations

from .data import Example, SpanLabel, iter_token_spans


def build_bio_labels(span_labels: list[str]) -> list[str]:
    labels = ["O"]
    for label in sorted(span_labels):
        labels.append(f"B-{label}")
        labels.append(f"I-{label}")
    return labels


def spans_to_bio(example: Example, clamp_offsets: bool = True) -> tuple[list[str], list[tuple[int, int, str]]]:
    tokens = iter_token_spans(example.text)
    tags = ["O"] * len(tokens)
    spans = [label.clamped(example.text) if clamp_offsets else label for label in example.labels]

    for span in spans:
        if span.start >= span.end:
            continue
        covered = [i for i, (start, end, _) in enumerate(tokens) if start < span.end and end > span.start]
        for pos, token_idx in enumerate(covered):
            prefix = "B" if pos == 0 else "I"
            tags[token_idx] = f"{prefix}-{span.label}"
    return tags, tokens


def bio_to_spans(tags: list[str], token_spans: list[tuple[int, int, str]]) -> list[SpanLabel]:
    spans: list[SpanLabel] = []
    current_label: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    def close_current() -> None:
        nonlocal current_label, current_start, current_end
        if current_label is not None and current_start is not None and current_end is not None:
            spans.append(SpanLabel(current_start, current_end, current_label))
        current_label = None
        current_start = None
        current_end = None

    for tag, (start, end, _) in zip(tags, token_spans):
        if tag == "O" or "-" not in tag:
            close_current()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or label != current_label:
            close_current()
            current_label = label
            current_start = start
            current_end = end
        else:
            current_end = end

    close_current()
    return spans
