"""Varlık düzeyinde (entity-level) P/R/F1: conlleval davranışı, bağımlılıksız."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def bio_spans(tags: Sequence[str]) -> set[tuple[str, int, int]]:
    """BIO etiketlerinden (tür, başlangıç, bitiş) kümeleri; B'siz I yeni varlık başlatır."""
    spans: set[tuple[str, int, int]] = set()
    cur: tuple[str, int] | None = None
    for i, t in enumerate([*tags, "O"]):
        kind = t[2:] if t != "O" else ""
        starts = t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != kind))
        if cur is not None and (t == "O" or starts):
            spans.add((cur[0], cur[1], i - 1))
            cur = None
        if starts:
            cur = (kind, i)
    return spans


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def entity_f1(
    gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]
) -> dict[str, dict[str, float]]:
    """Tür bazında ve mikro/makro varlık F1."""
    if len(gold) != len(pred):
        raise ValueError("gold ve pred aynı sayıda dizi içermeli")
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for g, p in zip(gold, pred, strict=True):
        if len(g) != len(p):
            raise ValueError("dizi uzunlukları eşleşmiyor")
        gs, ps = bio_spans(g), bio_spans(p)
        for s in gs & ps:
            tp[s[0]] += 1
        for s in ps - gs:
            fp[s[0]] += 1
        for s in gs - ps:
            fn[s[0]] += 1
    kinds = sorted(set(tp) | set(fp) | set(fn))
    out = {k: prf(tp[k], fp[k], fn[k]) for k in kinds}
    out["micro"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    out["macro"] = {
        m: round(sum(out[k][m] for k in kinds) / len(kinds), 4) if kinds else 0.0
        for m in ("precision", "recall", "f1")
    }
    return out
