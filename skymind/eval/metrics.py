"""Planlayıcı çıktısı için tek tip metrikler.

Her örnek: model çıktısı (ham metin) + altın çıktı (sözlük). Metrikler:

* ``json_valid``       – çıktı JSON mu ve şemaya uyuyor mu
* ``type_accuracy``    – plan/clarification/refusal tipi doğru mu
* ``action_accuracy``  – plan tipinde aksiyon dizisi birebir doğru mu
* ``field_accuracy``   – plan tipinde, aksiyon başına alanların toleranslı eşleşmesi
* ``fabrication_rate`` – altın çıktı clarification/refusal iken modelin *plan* üretmesi
                         (eksik bilgiyi uydurma = halüsinasyonun ana ölçüsü)
* ``invalid_action_rate`` – şemada olmayan aksiyon/alan üretimi (JSON olarak ayrıştırılabilen
                         ama doğrulamayı geçemeyen çıktılar içinde)
* ``over_clarification_rate`` – altın çıktı plan iken modelin gereksiz soru sorması
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from skymind.schemas import Clarification, MissionPlan, Refusal, parse_output

REL_TOL = 0.01  # sayısal alanlarda göreli tolerans (%1)

Parsed = MissionPlan | Clarification | Refusal


@dataclass(frozen=True)
class Example:
    prediction: str
    gold: Mapping[str, Any]


@dataclass
class Report:
    n: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {"n": float(self.n), **self.metrics}


def _try_parse(raw: str) -> tuple[Parsed | None, bool]:
    """(ayrıştırılmış çıktı | None, ham metin geçerli JSON nesnesi mi)."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, False
    if not isinstance(obj, dict):
        return None, False
    try:
        return parse_output(obj), True
    except ValidationError:
        return None, True


def _close(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-9)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _flat(d: object, prefix: str = "") -> dict[str, object]:
    """İç içe yapıyı yaprak alanlara düzleştirir (alan bazlı doğruluk için)."""
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flat(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = d
    return out


def _field_score(pred_step: dict[str, Any], gold_step: dict[str, Any]) -> float:
    g = _flat(gold_step)
    p = _flat(pred_step)
    if not g:
        return 1.0
    return sum(1 for k, v in g.items() if k in p and _close(p[k], v)) / len(g)


def evaluate(examples: Iterable[Example]) -> Report:
    n = valid = type_ok = fabricated = over_clar = invalid_action = 0
    n_underspec = n_plan_gold = n_plan_pred_ok = action_ok = 0
    field_scores: list[float] = []
    json_parsable = 0

    for ex in examples:
        n += 1
        gold_type = ex.gold["type"]
        pred, parsable = _try_parse(ex.prediction)
        json_parsable += parsable
        valid += pred is not None
        if parsable and pred is None:
            invalid_action += 1

        if gold_type in ("clarification", "refusal"):
            n_underspec += 1
            fabricated += isinstance(pred, MissionPlan)
        else:
            n_plan_gold += 1
            over_clar += isinstance(pred, Clarification)

        if pred is not None and pred.type == gold_type:
            type_ok += 1
            if gold_type == "plan" and isinstance(pred, MissionPlan):
                n_plan_pred_ok += 1
                p_steps = [s.model_dump(mode="json") for s in pred.steps]
                g_steps = list(ex.gold["steps"])
                if [s["action"] for s in p_steps] == [s["action"] for s in g_steps]:
                    action_ok += 1
                    field_scores.append(
                        sum(_field_score(p, g) for p, g in zip(p_steps, g_steps, strict=True))
                        / len(g_steps)
                    )
                else:
                    field_scores.append(0.0)
        elif gold_type == "plan":
            field_scores.append(0.0)

    def rate(num: float, den: float) -> float:
        return num / den if den else float("nan")

    return Report(
        n=n,
        metrics={
            "json_parsable": rate(json_parsable, n),
            "json_valid": rate(valid, n),
            "type_accuracy": rate(type_ok, n),
            "action_accuracy": rate(action_ok, n_plan_gold),
            "field_accuracy": rate(sum(field_scores), n_plan_gold),
            "fabrication_rate": rate(fabricated, n_underspec),
            "over_clarification_rate": rate(over_clar, n_plan_gold),
            "invalid_action_rate": rate(invalid_action, n),
        },
    )
