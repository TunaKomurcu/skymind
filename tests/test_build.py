import json
from pathlib import Path

from skymind.data.build import (
    assign_splits,
    build,
    expand,
    gold_candidates,
    group_key,
    split_of,
)
from skymind.data.generator import generate


def test_same_plan_never_crosses_splits() -> None:
    rows = expand(generate(3000, seed=1), {})
    buckets = assign_splits(rows)
    where: dict[str, str] = {}
    for s, items in buckets.items():
        for r in items:
            if r["category"] == "plan":
                k = group_key(r)
                assert where.setdefault(k, s) == s


def test_paraphrase_stays_with_its_template_and_no_text_leak() -> None:
    base = generate(600, seed=2)
    paras = {r["id"]: [r["command"] + " lütfen yap", r["command"] + " hemen"] for r in base[:300]}
    rows = expand(base, paras)
    buckets = assign_splits(rows)
    split_by_group: dict[str, set[str]] = {}
    for s, items in buckets.items():
        for r in items:
            if r["category"] != "plan":
                split_by_group.setdefault(r["group"], set()).add(s)
    assert all(len(v) == 1 for v in split_by_group.values())
    texts = [r["command"].lower().strip() for items in buckets.values() for r in items]
    assert len(texts) == len(set(texts))


def test_split_proportions() -> None:
    buckets = assign_splits(expand(generate(4000, seed=3), {}))
    n = sum(len(v) for v in buckets.values())
    assert 0.65 < len(buckets["train"]) / n < 0.92
    assert len(buckets["test"]) > 0 and len(buckets["val"]) > 0
    assert split_of("x") == split_of("x")


def test_gold_candidates_are_template_only_and_stratified() -> None:
    buckets = assign_splits(expand(generate(6000, seed=4), {}))
    gold = gold_candidates(buckets["test"])
    assert 250 <= len(gold) <= 300
    assert {g["source"] for g in gold} == {"template"}
    assert {"plan", "clarification", "refusal"} <= {g["category"] for g in gold}


def test_build_writes_files(tmp_path: Path) -> None:
    para = tmp_path / "p.jsonl"
    para.write_text("", encoding="utf-8")
    stats = build(500, 5, para, tmp_path / "out", tmp_path / "gold.jsonl")
    assert (tmp_path / "out" / "train.jsonl").exists()
    assert json.loads((tmp_path / "out" / "stats.json").read_text("utf-8")) == stats
