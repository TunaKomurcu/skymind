import json

from skymind.data import generate
from skymind.planner.baselines import BASELINES, run_baseline
from skymind.planner.prompt import SYSTEM_PROMPT, dpo_example, messages, sft_example
from skymind.schemas import ACTION_NAMES, parse_output

ROWS = generate(400, seed=9)


def test_system_prompt_mentions_every_action() -> None:
    for a in ACTION_NAMES:
        assert a in SYSTEM_PROMPT


def test_sft_is_prompt_completion_with_valid_compact_json() -> None:
    for r in ROWS:
        ex = sft_example(r)
        assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
        assert [m["role"] for m in ex["completion"]] == ["assistant"]
        parse_output(json.loads(ex["completion"][0]["content"]))
    assert len(messages("x")) == 2


def test_dpo_pairs_only_for_underspecified_and_differ() -> None:
    for r in ROWS:
        ex = dpo_example(r)
        if r["category"] == "plan":
            assert ex is None
        else:
            assert ex is not None
            assert ex["chosen"] != ex["rejected"]
            assert json.loads(ex["rejected"][0]["content"])["type"] == "plan"


def test_baselines_bracket_the_metrics() -> None:
    plan = run_baseline(BASELINES["always_plan"], ROWS).metrics
    assert plan["fabrication_rate"] == 1.0  # tavan: eksik/ret durumlarının hepsinde uyduruyor
    assert plan["json_valid"] == 1.0
    junk = run_baseline(BASELINES["not_json"], ROWS).metrics
    assert junk["json_valid"] == 0.0
    refuse = run_baseline(BASELINES["always_refuse"], ROWS).metrics
    assert refuse["fabrication_rate"] == 0.0 and refuse["action_accuracy"] == 0.0
