"""Model gerektirmeyen taban çizgileri: metriklerin alt sınırını gösterir."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from skymind.eval import Example, Report, evaluate
from skymind.planner.prompt import to_json

Baseline = Callable[[str], str]


def always_plan(_command: str) -> str:
    """Her şeye makul görünen bir plan döndürür (halüsinasyon tavanı)."""
    return to_json(
        {
            "type": "plan",
            "steps": [{"action": "goto", "target": {"kind": "named", "name": "Alfa noktası"}}],
        }
    )


def always_clarify(_command: str) -> str:
    return to_json(
        {"type": "clarification", "question": "Hangi bilgiyi istiyorsunuz?", "missing": ["x"]}
    )


def always_refuse(_command: str) -> str:
    return to_json({"type": "refusal", "reason": "Yapamam."})


def not_json(_command: str) -> str:
    return "Tamam, hemen yapıyorum."


BASELINES: dict[str, Baseline] = {
    "always_plan": always_plan,
    "always_clarify": always_clarify,
    "always_refuse": always_refuse,
    "not_json": not_json,
}


def run_baseline(fn: Baseline, rows: list[dict[str, Any]]) -> Report:
    return evaluate(Example(prediction=fn(r["command"]), gold=r["gold"]) for r in rows)


def pretty(report: Report) -> str:
    return json.dumps({k: round(v, 3) for k, v in report.as_dict().items()}, ensure_ascii=False)
