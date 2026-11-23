import json

import jsonschema
import pytest
from pydantic import ValidationError

from skymind.schemas import Clarification, MissionPlan, Refusal, json_schema, parse_output

SCAN = {
    "type": "plan",
    "steps": [
        {
            "action": "scan",
            "center": {"kind": "relative", "direction": "N", "distance_m": 3000},
            "radius_m": 3000,
            "altitude_m": 1500,
        }
    ],
}


def test_valid_plan_roundtrip() -> None:
    out = parse_output(json.dumps(SCAN))
    assert isinstance(out, MissionPlan)
    assert out.steps[0].action == "scan"


def test_clarification_and_refusal() -> None:
    c = parse_output(
        {"type": "clarification", "question": "Hangi irtifada?", "missing": ["altitude"]}
    )
    r = parse_output({"type": "refusal", "reason": "Silah kullanımı kapsam dışı."})
    assert isinstance(c, Clarification)
    assert isinstance(r, Refusal)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["steps"][0].update(altitude_m=99999),  # sınır dışı irtifa
        lambda d: d["steps"][0].update(radius_m=1),  # çok küçük yarıçap
        lambda d: d["steps"][0].update(action="bomb"),  # şemada olmayan aksiyon
        lambda d: d["steps"][0].update(extra_field=1),  # fazladan alan
        lambda d: d.update(type="unknown"),
        lambda d: d.update(steps=[]),
    ],
)
def test_invalid_plans_rejected(mutate) -> None:  # type: ignore[no-untyped-def]
    data = json.loads(json.dumps(SCAN))
    mutate(data)
    with pytest.raises(ValidationError):
        parse_output(data)


def test_clarification_requires_missing() -> None:
    with pytest.raises(ValidationError):
        parse_output({"type": "clarification", "question": "Nerede?", "missing": []})


def test_exported_json_schema_agrees_with_pydantic() -> None:
    schema = json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(SCAN, schema)
    bad = json.loads(json.dumps(SCAN))
    bad["steps"][0]["altitude_m"] = 99999
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
