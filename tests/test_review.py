"""Altın küme inceleme aracı: karar kaydı, yeniden başlatma, geri alma, düzeltme, dışa aktarma."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from skymind.data import review

PLAN = {"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 120.0}]}
CLAR = {"type": "clarification", "question": "Hedef nedir?", "missing": ["center"]}
REF = {"type": "refusal", "reason": "Silah kullanımı kapsam dışı."}


def make_candidates(path: Path) -> list[dict[str, object]]:
    rows = [
        {"id": "a", "command": "120 metreye kalk", "gold": PLAN, "category": "plan", "group": "a"},
        {
            "id": "b",
            "command": "daire çiz",
            "gold": CLAR,
            "category": "clarification",
            "group": "b",
        },
        {"id": "c", "command": "bomba bırak", "gold": REF, "category": "refusal", "group": "c"},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
    return rows


def feed(monkeypatch: pytest.MonkeyPatch, keys: list[str]) -> None:
    it: Iterator[str] = iter(keys)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it, "q"))


def test_render_gold_covers_all_three_output_types() -> None:
    assert "1. takeoff" in review.render_gold(PLAN)
    assert "EKSİK ALANLAR: center" in review.render_gold(CLAR)
    assert "RET GEREKÇESİ" in review.render_gold(REF)


def test_validate_gold_accepts_schema_valid_and_rejects_the_rest() -> None:
    assert review.validate_gold(json.dumps(PLAN))["type"] == "plan"
    for bad in ("kod değil", '{"type":"plan","steps":[{"action":"takeoff","altitude_m":9999}]}'):
        with pytest.raises(ValueError):
            review.validate_gold(bad)


def test_review_flow_resume_undo_edit_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src, dec, out = tmp_path / "c.jsonl", tmp_path / "d.jsonl", tmp_path / "gold.jsonl"
    make_candidates(src)
    order = [c["id"] for c in review.load_candidates(src)]
    # sıra: 1. aday doğru (Enter), 2. aday hatalı (n), sonra çık
    feed(monkeypatch, ["", "n", "3", "q"])  # n sonrası neden sorulur: 3 = gereksiz soru
    review.run(src, dec)
    d = review.load_decisions(dec)
    assert d[order[0]]["status"] == "ok" and d[order[1]]["status"] == "bad"
    assert d[order[1]]["reason"] == "gereksiz_soru" and d[order[0]]["reason"] == ""
    assert review.reason_counts(dec) == {"gereksiz_soru": 1}
    # yeniden başlat: yalnızca 3. aday kalmış olmalı; komutu düzelt (c) ve onayla -> edited
    feed(monkeypatch, ["c", "yeni komut metni", "", "q"])
    review.run(src, dec)
    d = review.load_decisions(dec)
    assert d[order[2]]["status"] == "edited" and d[order[2]]["command"] == "yeni komut metni"
    # geri al: son karar silinir
    feed(monkeypatch, ["b", "q"])
    review.run(src, dec)
    # geri al yalnızca bu oturumdaki kararları siler; bu oturumda karar yok -> değişmez
    assert len(review.load_decisions(dec)) == 3
    # dışa aktar: yalnızca ok + edited, sayılar yeni komuttan yeniden hesaplanır
    res = review.export(src, dec, out)
    assert res["exported"] == 2 and res["bad"] == 1 and res["pending"] == 0
    rows = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert {r["id"] for r in rows} == {order[0], order[2]}
    assert all(r["source"] == "gold-reviewed" and r["category"] == r["gold"]["type"] for r in rows)


def test_undo_in_session_and_gold_edit_with_invalid_then_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, dec = tmp_path / "c.jsonl", tmp_path / "d.jsonl"
    make_candidates(src)
    first = review.load_candidates(src)[0]
    new_gold = json.dumps({"type": "plan", "steps": [{"action": "takeoff", "altitude_m": 150.0}]})
    # g: geçersiz JSON reddedilir, sonra geçerli kabul edilir; Enter -> edited; b geri alır; q
    feed(monkeypatch, ["g", "saçma", "g", new_gold, "", "b", "q"])
    review.run(src, dec)
    assert review.load_decisions(dec) == {}  # düzeltme geri alındı, hiçbir karar kalmadı
    feed(monkeypatch, ["g", new_gold, "", "q"])
    review.run(src, dec)
    d = review.load_decisions(dec)[first["id"]]
    assert d["status"] == "edited" and d["gold"]["steps"][0]["altitude_m"] == 150.0


def test_skip_defers_the_candidate_and_stats_count_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, dec = tmp_path / "c.jsonl", tmp_path / "d.jsonl"
    make_candidates(src)
    feed(monkeypatch, ["s", "", "q"])  # 1. atla, 2. onayla
    review.run(src, dec)
    s = review.stats(src, dec)
    assert s == {"total": 3, "ok": 1, "edited": 0, "bad": 0, "pending": 2}


def test_candidate_order_is_deterministic_and_covers_every_candidate(tmp_path: Path) -> None:
    src = tmp_path / "c.jsonl"
    make_candidates(src)
    a = [c["id"] for c in review.load_candidates(src, seed=0)]
    assert a == [c["id"] for c in review.load_candidates(src, seed=0)]
    assert sorted(a) == ["a", "b", "c"]
