from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data import Example, SpanLabel, read_jsonl, split_aspect_polarity, write_jsonl


ASPECT_KEYWORDS = {
    "BATTERY": ["pin", "sạc", "chai", "tụt"],
    "CAMERA": ["camera", "cam", "chụp", "ảnh"],
    "PERFORMANCE": ["lag", "giật", "mượt", "chip", "game", "ram", "đơ", "treo"],
    "SCREEN": ["màn hình", "màn", "hiển thị"],
    "DESIGN": ["đẹp", "thiết kế", "cầm", "mỏng", "nặng", "nhẹ"],
    "PRICE": ["giá", "tiền", "rẻ", "đắt"],
    "FEATURES": ["vân tay", "loa", "face", "tính năng", "wifi", "sóng"],
    "SER&ACC": ["nhân viên", "tgdd", "phục vụ", "bảo hành", "shop", "giao hàng"],
    "STORAGE": ["bộ nhớ", "dung lượng", "lưu trữ"],
}

POSITIVE_WORDS = [
    "tốt",
    "ổn",
    "ok",
    "mượt",
    "đẹp",
    "nhanh",
    "trâu",
    "nét",
    "hài lòng",
    "nhiệt tình",
    "rẻ",
]
NEGATIVE_WORDS = [
    "tệ",
    "kém",
    "chậm",
    "lag",
    "giật",
    "đơ",
    "nóng",
    "tụt",
    "hao",
    "lỗi",
    "không",
    "ko",
    "quá",
    "đắt",
]


def compile_patterns(words: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(r"(?iu)(?<!\w)" + re.escape(word) + r"(?!\w)") for word in words]


ASPECT_PATTERNS = {aspect: compile_patterns(words) for aspect, words in ASPECT_KEYWORDS.items()}
POS_PATTERNS = compile_patterns(POSITIVE_WORDS)
NEG_PATTERNS = compile_patterns(NEGATIVE_WORDS)


def infer_default_polarity(train_examples: list[Example]) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for ex in train_examples:
        for label in ex.labels:
            aspect, polarity = split_aspect_polarity(label.label)
            counts[aspect][polarity] += 1
    return {aspect: counter.most_common(1)[0][0] for aspect, counter in counts.items()}


def score_polarity(window: str, default: str) -> str:
    pos = sum(1 for pattern in POS_PATTERNS if pattern.search(window))
    neg = sum(1 for pattern in NEG_PATTERNS if pattern.search(window))
    if pos > neg:
        return "POSITIVE"
    if neg > pos:
        return "NEGATIVE"
    return default


def predict_example(text: str, defaults: dict[str, str]) -> list[SpanLabel]:
    predictions: list[SpanLabel] = []
    seen: set[tuple[int, int, str]] = set()
    lower_text = text.lower()

    for aspect, patterns in ASPECT_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(lower_text):
                start, end = match.span()
                window = lower_text[max(0, start - 35) : min(len(text), end + 35)]
                polarity = score_polarity(window, defaults.get(aspect, "POSITIVE"))
                label = f"{aspect}#{polarity}"
                key = (start, end, label)
                if key not in seen:
                    predictions.append(SpanLabel(start, end, label))
                    seen.add(key)

    if not predictions:
        positive = any(pattern.search(lower_text) for pattern in POS_PATTERNS)
        negative = any(pattern.search(lower_text) for pattern in NEG_PATTERNS)
        if positive or negative:
            polarity = "POSITIVE" if positive and not negative else "NEGATIVE"
            predictions.append(SpanLabel(0, min(len(text), 40), f"GENERAL#{polarity}"))

    return sorted(predictions, key=lambda span: (span.start, span.end, span.label))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple keyword ABSA baseline.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_examples = read_jsonl(data_dir / "train.jsonl")
    examples = read_jsonl(data_dir / f"{args.split}.jsonl")
    defaults = infer_default_polarity(train_examples)

    rows = []
    for ex in examples:
        labels = [[span.start, span.end, span.label] for span in predict_example(ex.text, defaults)]
        rows.append({"text": ex.text, "labels": labels})
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
