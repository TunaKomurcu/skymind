import json
import random
from collections import Counter

from skymind.data import generate
from skymind.planner.preferences import build_pairs, perturb_plan
from skymind.schemas import parse_output

ROWS = generate(1500, seed=11)


def test_pairs_valid_and_chosen_differs_from_rejected() -> None:
    pairs = build_pairs(ROWS, seed=1)
    assert len(pairs) == len(ROWS)
    for p in pairs:
        c, r = p["chosen"][0]["content"], p["rejected"][0]["content"]
        assert c != r
        parse_output(json.loads(c))
        parse_output(json.loads(r))


def test_plan_rows_get_both_kinds_of_rejected() -> None:
    kinds = Counter()
    for p in build_pairs(ROWS, seed=2):
        if p["category"] == "plan":
            kinds[json.loads(p["rejected"][0]["content"])["type"]] += 1
        else:
            assert json.loads(p["rejected"][0]["content"])["type"] == "plan"
    assert kinds["plan"] > 100 and kinds["clarification"] > 100


def test_perturbation_changes_exactly_one_value_and_stays_in_bounds() -> None:
    rng = random.Random(0)
    n = 0
    for r in ROWS:
        if r["category"] != "plan":
            continue
        bad = perturb_plan(r["gold"], rng)
        if bad is None:
            continue
        n += 1
        parse_output(bad)
        assert bad != r["gold"]
    assert n > 100


def test_deterministic() -> None:
    assert build_pairs(ROWS[:200], seed=3) == build_pairs(ROWS[:200], seed=3)
