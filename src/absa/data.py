from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class SpanLabel:
    start: int
    end: int
    label: str

    def clamped(self, text: str) -> "SpanLabel":
        start = max(0, min(self.start, len(text)))
        end = max(start, min(self.end, len(text)))
        return SpanLabel(start, end, self.label)


@dataclass(frozen=True)
class Example:
    text: str
    labels: list[SpanLabel]


def read_jsonl(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels = [SpanLabel(int(a), int(b), str(label)) for a, b, label in row.get("labels", [])]
            examples.append(Example(text=str(row["text"]), labels=labels))
    return examples


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_token_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(text)]


def label_inventory(examples: Iterable[Example]) -> list[str]:
    return sorted({label.label for ex in examples for label in ex.labels})


def split_aspect_polarity(label: str) -> tuple[str, str]:
    if "#" not in label:
        return label, ""
    aspect, polarity = label.rsplit("#", 1)
    return aspect, polarity
