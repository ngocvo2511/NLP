from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .data import Example, SpanLabel, split_aspect_polarity


def exact_key(span: SpanLabel, text: str, clamp_offsets: bool = True) -> tuple[int, int, str]:
    if clamp_offsets:
        span = span.clamped(text)
    return span.start, span.end, span.label


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_exact(gold: Iterable[Example], pred: Iterable[Example]) -> dict:
    total_tp = total_fp = total_fn = 0
    by_label: dict[str, Counter] = defaultdict(Counter)
    by_aspect: dict[str, Counter] = defaultdict(Counter)
    by_polarity: dict[str, Counter] = defaultdict(Counter)

    for gold_ex, pred_ex in zip(gold, pred):
        gold_set = {exact_key(span, gold_ex.text) for span in gold_ex.labels}
        pred_set = {exact_key(span, gold_ex.text) for span in pred_ex.labels}

        tp_items = gold_set & pred_set
        fp_items = pred_set - gold_set
        fn_items = gold_set - pred_set

        total_tp += len(tp_items)
        total_fp += len(fp_items)
        total_fn += len(fn_items)

        for bucket_name, items in (("tp", tp_items), ("fp", fp_items), ("fn", fn_items)):
            for _, _, label in items:
                aspect, polarity = split_aspect_polarity(label)
                by_label[label][bucket_name] += 1
                by_aspect[aspect][bucket_name] += 1
                by_polarity[polarity][bucket_name] += 1

    result = {
        "micro": {"tp": total_tp, "fp": total_fp, "fn": total_fn, **prf(total_tp, total_fp, total_fn)},
        "by_label": summarize_counters(by_label),
        "by_aspect": summarize_counters(by_aspect),
        "by_polarity": summarize_counters(by_polarity),
    }
    return result


def summarize_counters(counters: dict[str, Counter]) -> dict[str, dict[str, float]]:
    summary = {}
    for name, counts in sorted(counters.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        summary[name] = {"tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)}
    return summary
