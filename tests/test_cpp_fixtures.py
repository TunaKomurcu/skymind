"""C++ servisinin dayandığı sabit dosyalar Python kaynaklarıyla senkron kalmalı."""

import json
from pathlib import Path

from pydantic import ValidationError

from skymind.planner.prompt import SYSTEM_SHORT
from skymind.schemas import parse_output

CPP = Path(__file__).resolve().parents[1] / "cpp"


def test_cpp_system_prompt_matches_training_prompt() -> None:
    text = (CPP / "configs" / "system_short.txt").read_text(encoding="utf-8")
    assert text.rstrip("\n") == SYSTEM_SHORT, "python scripts/export_cpp_fixtures.py çalıştırın"


def test_parity_cases_match_pydantic_and_cover_both_outcomes() -> None:
    lines = (CPP / "tests" / "data" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line]
    assert len(cases) > 200
    for c in cases:
        try:
            parse_output(c["doc"])
            ok = True
        except ValidationError:
            ok = False
        assert ok == c["valid"], c["doc"]
    assert sum(c["valid"] for c in cases) > 50
    assert sum(not c["valid"] for c in cases) > 50
