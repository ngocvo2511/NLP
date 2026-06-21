from __future__ import annotations

from .data import SpanLabel


def merge_nearby_same_label(spans: list[SpanLabel], max_gap: int) -> list[SpanLabel]:
    if max_gap < 0:
        return sorted(spans, key=lambda span: (span.start, span.end, span.label))

    merged: list[SpanLabel] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label)):
        if (
            merged
            and merged[-1].label == span.label
            and span.start >= merged[-1].end
            and span.start - merged[-1].end <= max_gap
        ):
            previous = merged[-1]
            merged[-1] = SpanLabel(previous.start, max(previous.end, span.end), previous.label)
        else:
            merged.append(span)
    return merged


def filter_short_spans(spans: list[SpanLabel], min_chars: int) -> list[SpanLabel]:
    return [span for span in spans if span.end - span.start >= min_chars]


def postprocess_spans(
    spans: list[SpanLabel],
    text: str,
    min_chars: int = 1,
    merge_gap: int = -1,
) -> list[SpanLabel]:
    valid = [span.clamped(text) for span in spans]
    valid = [span for span in valid if span.start < span.end]
    valid = merge_nearby_same_label(valid, merge_gap)
    valid = filter_short_spans(valid, min_chars)

    seen: set[tuple[int, int, str]] = set()
    deduped: list[SpanLabel] = []
    for span in sorted(valid, key=lambda item: (item.start, item.end, item.label)):
        key = (span.start, span.end, span.label)
        if key not in seen:
            deduped.append(span)
            seen.add(key)
    return deduped
