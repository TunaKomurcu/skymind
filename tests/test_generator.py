from collections import Counter

from skymind.data import generate
from skymind.data.generator import numbers_in, tr_upper_first
from skymind.schemas import MissionPlan, parse_output


def test_deterministic_and_unique() -> None:
    a, b = generate(300, seed=7), generate(300, seed=7)
    assert a == b
    assert generate(300, seed=8) != a
    cmds = [r["command"].lower() for r in a]
    assert len(set(cmds)) == len(cmds)


def test_all_gold_and_rejected_valid() -> None:
    for r in generate(1500, seed=1):
        parse_output(r["gold"])
        if "rejected" in r:
            assert isinstance(parse_output(r["rejected"]), MissionPlan)


def test_category_mix_roughly_matches_weights() -> None:
    c = Counter(r["category"] for r in generate(2000, seed=2))
    assert 0.5 < c["plan"] / 2000 < 0.7
    assert c["clarification"] > 300 and c["refusal"] > 150


def test_clarification_and_refusal_always_have_rejected_plan() -> None:
    for r in generate(500, seed=3):
        if r["category"] == "plan":
            assert "rejected" not in r
        else:
            assert r["rejected"]["type"] == "plan"
            assert r["rejected"] != r["gold"]


def test_plan_commands_contain_their_numbers() -> None:
    """Plan örneklerinde komuttaki sayılar altın plandaki değerlerle tutarlı olmalı."""
    for r in generate(800, seed=4):
        if r["category"] != "plan":
            continue
        text_nums = {float(x) for x in numbers_in(r["command"])}
        for s in r["gold"]["steps"]:
            for key in ("altitude_m", "radius_m"):
                if key in s:
                    v = s[key]
                    assert v in text_nums or v / 1000 in text_nums, (r["command"], key, v)


def test_clarification_commands_omit_the_missing_slot() -> None:
    words = {"altitude_m": ("irtifa", "yüksekli"), "radius_m": ("yarıçap",)}
    for r in generate(800, seed=5):
        if r["category"] != "clarification":
            continue
        for field, stems in words.items():
            if field in r["gold"]["missing"]:
                assert not any(w in r["command"].lower() for w in stems), r["command"]


def test_turkish_capitalisation() -> None:
    assert tr_upper_first("iniş yap") == "İniş yap"
    assert tr_upper_first("ıslak") == "Islak"
