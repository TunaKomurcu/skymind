"""Niyet ve NER veri kümeleri (işlenmiş bölümlerden) ve taban çizgileri."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from skymind.nlp.ner_data import ENTITY_TYPES, INTENTS, align_example, intent_of, words

SPLITS = Path("data/processed")


def load(split: str, root: Path = SPLITS) -> list[dict[str, Any]]:
    path = root / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def intent_data(split: str, root: Path = SPLITS) -> tuple[list[str], list[int]]:
    rows = load(split, root)
    return [r["command"] for r in rows], [INTENTS.index(intent_of(r["gold"])) for r in rows]


def ner_data(split: str, root: Path = SPLITS) -> tuple[list[list[str]], list[list[str]]]:
    """(kelime listeleri, BIO etiket listeleri); yalnızca hizalanabilen plan örnekleri."""
    words: list[list[str]] = []
    tags: list[list[str]] = []
    for r in load(split, root):
        if r["gold"]["type"] != "plan":
            continue
        al = align_example(r["command"], r["gold"])
        if al is not None:
            words.append(al[0])
            tags.append(al[1])
    return words, tags


MANUAL = Path("data/gold/ner_manual.txt")
_MARK = re.compile(r"\[([^\]|]+)\|([A-Z]+)\]")


def parse_marked(line: str) -> tuple[list[str], list[str]]:
    """``"[500 m|ALT] irtifaya çık"`` -> (kelimeler, BIO etiketleri)."""
    plain, spans, pos = "", [], 0
    for m in _MARK.finditer(line):
        plain += line[pos : m.start()]
        spans.append((len(plain), len(plain) + len(m.group(1)), m.group(2)))
        plain += m.group(1)
        pos = m.end()
    plain += line[pos:]
    if "[" in plain or "|" in plain:
        raise ValueError(f"bozuk işaretleme: {line!r}")
    ws = words(plain)
    tags = ["O"] * len(ws)
    for s, e, label in spans:
        if label not in ENTITY_TYPES:
            raise ValueError(f"bilinmeyen etiket {label!r}: {line!r}")
        idx = [i for i, (_w, ws_, we_) in enumerate(ws) if ws_ < e and we_ > s]
        if not idx:
            raise ValueError(f"aralık kelimeye denk gelmedi: {line!r}")
        tags[idx[0]] = f"B-{label}"
        for i in idx[1:]:
            tags[i] = f"I-{label}"
    return [w[0] for w in ws], tags


def manual_ner(path: Path = MANUAL) -> dict[str, tuple[list[list[str]], list[list[str]]]]:
    """Bölümlere (KOLAY/ZOR) göre (kelimeler, etiketler)."""
    out: dict[str, tuple[list[list[str]], list[list[str]]]] = {}
    section = "KOLAY"
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("##"):
            section = raw.strip("# ").strip()
        elif raw.strip() and not raw.startswith("#"):
            w, t = parse_marked(raw)
            ws, ts = out.setdefault(section, ([], []))
            ws.append(w)
            ts.append(t)
    return out
