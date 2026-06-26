from __future__ import annotations

from .data import Example, SpanLabel, iter_token_spans


def build_sequence_labels(span_labels: list[str], scheme: str = "bio") -> list[str]:
    if scheme not in {"bio", "bilou"}:
        raise ValueError(f"Unsupported tag scheme: {scheme}")

    labels = ["O"]
    prefixes = ["B", "I"] if scheme == "bio" else ["B", "I", "L", "U"]
    for label in sorted(span_labels):
        labels.extend(f"{prefix}-{label}" for prefix in prefixes)
    return labels


def build_bio_labels(span_labels: list[str]) -> list[str]:
    return build_sequence_labels(span_labels, scheme="bio")


def tag_prefix(position: int, length: int, scheme: str) -> str:
    if scheme == "bio":
        return "B" if position == 0 else "I"
    if scheme == "bilou":
        if length == 1:
            return "U"
        if position == 0:
            return "B"
        if position == length - 1:
            return "L"
        return "I"
    raise ValueError(f"Unsupported tag scheme: {scheme}")


def spans_to_tags(
    example: Example,
    clamp_offsets: bool = True,
    scheme: str = "bio",
) -> tuple[list[str], list[tuple[int, int, str]]]:
    tokens = iter_token_spans(example.text)
    return spans_to_tags_with_offsets(example, tokens, clamp_offsets=clamp_offsets, scheme=scheme)


def spans_to_tags_with_offsets(
    example: Example,
    offsets: list[tuple[int, int, str]],
    clamp_offsets: bool = True,
    scheme: str = "bio",
) -> tuple[list[str], list[tuple[int, int, str]]]:
    tags = ["O"] * len(offsets)
    spans = [label.clamped(example.text) if clamp_offsets else label for label in example.labels]

    for span in spans:
        if span.start >= span.end:
            continue
        covered = [i for i, (start, end, _) in enumerate(offsets) if start < span.end and end > span.start]
        for pos, token_idx in enumerate(covered):
            prefix = tag_prefix(pos, len(covered), scheme)
            tags[token_idx] = f"{prefix}-{span.label}"
    return tags, offsets


def spans_to_bio(example: Example, clamp_offsets: bool = True) -> tuple[list[str], list[tuple[int, int, str]]]:
    return spans_to_tags(example, clamp_offsets=clamp_offsets, scheme="bio")


def tags_to_spans(tags: list[str], token_spans: list[tuple[int, int, str]]) -> list[SpanLabel]:
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


def bio_to_spans(tags: list[str], token_spans: list[tuple[int, int, str]]) -> list[SpanLabel]:
    return tags_to_spans(tags, token_spans)
