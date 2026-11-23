"""DPO tercih çiftleri.

* ``clarification`` / ``refusal`` satırları: chosen = altın cevap, rejected = uydurma plan
  (üreteçten gelen ``rejected``).
* ``plan`` satırları: chosen = altın plan, rejected = (a) bir sayısal alanı bozulmuş plan ya da
  (b) gereksiz netleştirme sorusu. Böylece model "her şeye soru sor / reddet"e kaymaz.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from skymind.planner.prompt import SYSTEM_SHORT, messages, to_json
from skymind.schemas import parse_output
from skymind.schemas.plan import (
    MAX_ALT_M,
    MAX_RADIUS_M,
    MAX_SPEED_MS,
    MIN_ALT_M,
    MIN_RADIUS_M,
    MIN_SPEED_MS,
)

NUMERIC_BOUNDS = {
    "altitude_m": (MIN_ALT_M, MAX_ALT_M),
    "radius_m": (MIN_RADIUS_M, MAX_RADIUS_M),
    "speed_ms": (MIN_SPEED_MS, MAX_SPEED_MS),
}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E", "NE": "SW", "SW": "NE", "NW": "SE", "SE": "NW"}
NEEDLESS_QUESTIONS = [
    ("Hangi irtifada uçmamı istersiniz?", ["altitude_m"]),
    ("Hedef bölgeyi belirtir misiniz?", ["target"]),
    ("Lütfen yarıçap bilgisini verir misiniz?", ["radius_m"]),
    ("Komutu netleştirir misiniz?", ["command"]),
]


def perturb_plan(gold: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Şemaya uyan ama yanlış bir plan; bozulacak alan yoksa None."""
    out = copy.deepcopy(gold)
    slots: list[tuple[dict[str, Any], str]] = []
    for step in out["steps"]:
        slots += [(step, k) for k in NUMERIC_BOUNDS if k in step]
        loc = step.get("center") or step.get("target")
        if isinstance(loc, dict) and loc.get("kind") == "relative":
            slots.append((loc, "direction"))
    if not slots:
        return None
    holder, key = rng.choice(slots)
    if key == "direction":
        holder[key] = OPPOSITE[holder[key]]
    else:
        lo, hi = NUMERIC_BOUNDS[key]
        factor = rng.choice([0.5, 2.0, 3.0])
        new = min(max(holder[key] * factor, lo), hi)
        if new == holder[key]:
            new = max(lo, holder[key] / 2) if holder[key] > lo else min(hi, holder[key] * 2)
        holder[key] = float(new)
    return out


def rejected_for(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if row["category"] != "plan":
        return dict(row["rejected"])
    if rng.random() < 0.5:
        bad = perturb_plan(row["gold"], rng)
        if bad is not None:
            return bad
    q, missing = rng.choice(NEEDLESS_QUESTIONS)
    return {"type": "clarification", "question": q, "missing": missing}


def build_pairs(
    rows: list[dict[str, Any]], seed: int = 0, system: str = SYSTEM_SHORT
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for r in rows:
        rejected = rejected_for(r, rng)
        parse_output(rejected)  # reddedilen cevap da şema-geçerli olmalı: fark içerikte
        if rejected == r["gold"]:
            continue
        pairs.append(
            {
                "prompt": messages(r["command"], system=system),
                "chosen": [{"role": "assistant", "content": to_json(r["gold"])}],
                "rejected": [{"role": "assistant", "content": to_json(rejected)}],
                "category": r["category"],
            }
        )
    return pairs
