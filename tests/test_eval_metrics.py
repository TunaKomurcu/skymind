import json

import pytest

from skymind.eval import Example, evaluate

PLAN = {
    "type": "plan",
    "steps": [
        {"action": "takeoff", "altitude_m": 100},
        {
            "action": "scan",
            "center": {"kind": "relative", "direction": "N", "distance_m": 3000},
            "radius_m": 3000,
            "altitude_m": 1500,
        },
    ],
}
CLARIFY = {"type": "clarification", "question": "Hangi irtifada?", "missing": ["altitude"]}
REFUSE = {"type": "refusal", "reason": "Kapsam dışı."}


def ex(pred: object, gold: dict) -> Example:  # type: ignore[type-arg]
    return Example(prediction=pred if isinstance(pred, str) else json.dumps(pred), gold=gold)


def test_perfect() -> None:
    r = evaluate([ex(PLAN, PLAN), ex(CLARIFY, CLARIFY), ex(REFUSE, REFUSE)]).metrics
    assert r["json_valid"] == r["type_accuracy"] == r["action_accuracy"] == 1.0
    assert r["field_accuracy"] == 1.0
    assert r["fabrication_rate"] == 0.0


def test_fabrication_counted_only_on_underspecified() -> None:
    r = evaluate([ex(PLAN, CLARIFY), ex(PLAN, REFUSE), ex(CLARIFY, CLARIFY)]).metrics
    assert r["fabrication_rate"] == pytest.approx(2 / 3)


def test_over_clarification() -> None:
    r = evaluate([ex(CLARIFY, PLAN), ex(PLAN, PLAN)]).metrics
    assert r["over_clarification_rate"] == 0.5


def test_invalid_action_vs_non_json() -> None:
    bad = json.loads(json.dumps(PLAN))
    bad["steps"][0]["action"] = "bomb"
    r = evaluate([ex(bad, PLAN), ex("bu json değil", PLAN)]).metrics
    assert r["json_valid"] == 0.0
    assert r["json_parsable"] == 0.5
    assert r["invalid_action_rate"] == 0.5


def test_field_accuracy_tolerates_one_percent_and_penalizes_wrong_field() -> None:
    near = json.loads(json.dumps(PLAN))
    near["steps"][1]["radius_m"] = 3020  # %0.67 fark -> kabul
    assert evaluate([ex(near, PLAN)]).metrics["field_accuracy"] == 1.0
    wrong = json.loads(json.dumps(PLAN))
    wrong["steps"][1]["radius_m"] = 5000
    r = evaluate([ex(wrong, PLAN)]).metrics
    assert r["action_accuracy"] == 1.0
    assert r["field_accuracy"] < 1.0


def test_wrong_action_order_gets_zero_field_credit() -> None:
    swapped = {"type": "plan", "steps": list(reversed(PLAN["steps"]))}
    r = evaluate([ex(swapped, PLAN)]).metrics
    assert r["action_accuracy"] == 0.0
    assert r["field_accuracy"] == 0.0
