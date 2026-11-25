"""Hafif yardımcılar: deterministik alt küme seçimi ve yüzdelik dilim (ağır bağımlılık yok)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skymind.planner.prompt import load_split


def pick_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """planner.evaluate ile aynı deterministik, eşit aralıklı alt küme."""
    rows = load_split(path)
    if limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    return rows


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]
