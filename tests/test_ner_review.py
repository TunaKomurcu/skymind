"""NER elle set incelemesi: okuma, karar kaydı, doğrulama, uygulama ve HTTP uçları."""

import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from skymind.data import ner_review
from skymind.data.review import load_decisions
from skymind.data.review_web import make_handler
from skymind.nlp.datasets import manual_ner, parse_marked
from tests.test_review_web import call

REAL = Path("data/gold/ner_manual.txt")
SAMPLE = """# El yazımı mini NER sınama seti (3 cümle). Biçim: [metin|ETİKET].
# Yazar: Claude (asistan); proje sahibi gözden geçirmedi. Eğitim verisinden bağımsız.
## KOLAY
Lütfen [1500 metre|ALT] irtifaya çık.
[kuzeye|DIR] [2 km|DIST] ilerle.
## ZOR
[Alfa|PLACE] ve [Bravo|PLACE] noktaları arasında devriye at.
"""


@pytest.fixture
def state(tmp_path: Path) -> ner_review.NerState:
    src = tmp_path / "ner.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    return ner_review.NerState(src, tmp_path / "dec.jsonl")


def test_real_manual_set_has_exactly_40_sentences_in_two_sections() -> None:
    items = ner_review.load_items(REAL)
    assert len(items) == 40 and len({i["id"] for i in items}) == 40
    assert [sum(i["section"] == s for i in items) for s in ("KOLAY", "ZOR")] == [20, 20]
    sets = manual_ner(REAL)
    assert [len(sets[s][0]) for s in ("KOLAY", "ZOR")] == [20, 20]  # okuyucu aynı sayıyı görür


def test_decide_ok_edit_bad_and_validation(state: ner_review.NerState) -> None:
    a, b, c = state.items
    assert state.decide({"id": a["id"], "status": "ok", "markup": a["markup"]})["status"] == "ok"
    # cümle metni değişirse ('noktaları' iki kez) reddedilir
    changed = "[Alfa|PLACE] ve [Bravo noktaları|PLACE] noktaları arasında devriye at."
    with pytest.raises(ValueError, match="metni değiştirilemez"):
        state.decide({"id": c["id"], "status": "edited", "markup": changed})
    # yalnızca etiket sınırı değişirse kabul edilir ve "edited" olur
    alt = "[Alfa|PLACE] ve [Bravo noktaları|PLACE] arasında devriye at."
    assert ner_review.plain(alt) == ner_review.plain(c["markup"])
    assert state.decide({"id": c["id"], "status": "edited", "markup": alt})["status"] == "edited"
    assert state.decide({"id": b["id"], "status": "bad", "markup": b["markup"]})["status"] == "bad"
    for bad in (
        {"id": "yok", "status": "ok", "markup": a["markup"]},
        {"id": a["id"], "status": "belki", "markup": a["markup"]},
        {"id": a["id"], "status": "ok", "markup": "  "},
        {"id": a["id"], "status": "ok", "markup": "Lütfen [1500 metre|YOK] irtifaya çık."},
        {"id": a["id"], "status": "ok", "markup": "Lütfen [1500 metre|ALT irtifaya çık."},
    ):
        with pytest.raises(ValueError):
            state.decide(bad)
    saved = load_decisions(state.path)
    assert {d["status"] for d in saved.values()} == {"ok", "edited", "bad"}
    state.undo({"id": b["id"]})
    assert b["id"] not in load_decisions(state.path)


def test_apply_writes_edits_removes_bad_rewrites_header_and_keeps_a_backup(
    state: ner_review.NerState, tmp_path: Path
) -> None:
    a, b, c = state.items
    state.decide({"id": a["id"], "status": "ok", "markup": a["markup"]})
    state.decide({"id": b["id"], "status": "bad", "markup": b["markup"]})
    alt = "[Alfa|PLACE] ve [Bravo noktaları|PLACE] arasında devriye at."
    state.decide({"id": c["id"], "status": "edited", "markup": alt})
    src = tmp_path / "ner.txt"
    res = ner_review.apply(src, state.path)
    assert res["applied_edits"] == 1 and res["removed"] == 1 and res["sentences_now"] == 2
    text = src.read_text(encoding="utf-8")
    assert alt in text and "[kuzeye|DIR]" not in text
    assert "(2 cümle)" in text and "3/3 cümleyi gözden geçirdi (1 düzeltme, 1 çıkarma)" in text
    assert (tmp_path / "ner.bak.txt").read_text(encoding="utf-8") == SAMPLE  # yedek özgün hali
    for line in text.splitlines():  # yeni dosya hâlâ ayrıştırılabilir
        if line.strip() and not line.startswith("#"):
            parse_marked(line)


def test_stats_notes_and_untouched_file_when_nothing_changed(
    state: ner_review.NerState, tmp_path: Path
) -> None:
    assert ner_review.stats(tmp_path / "ner.txt", state.path) == {
        "total": 3, "ok": 0, "edited": 0, "bad": 0, "pending": 3,
    }  # fmt: skip
    src = tmp_path / "ner.txt"
    ner_review.apply(src, state.path)  # karar yok: satırlar aynı, yedek yok
    assert not (tmp_path / "ner.bak.txt").exists()
    assert "Lütfen [1500 metre|ALT]" in src.read_text(encoding="utf-8")
    real = ner_review.NerState(REAL, tmp_path / "x.jsonl")  # gerçek not dosyası okunur
    noted = [i for i in real.snapshot()["items"] if i["note"]]
    assert len(noted) == 3 and any("Alfa|PLACE] ve [Bravo|PLACE" in i["markup"] for i in noted)


@pytest.fixture
def server(state: ner_review.NerState) -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state, ner_review.PAGE))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_http_routes_serve_page_state_decide_and_refuse_foreign_routes(
    server: str, state: ner_review.NerState
) -> None:
    import urllib.request

    with urllib.request.urlopen(server + "/", timeout=5) as r:
        assert r.status == 200 and "NER" in r.read().decode("utf-8")
    snap = call(server + "/api/state")[1]
    assert len(snap["items"]) == 3 and set(snap["items"][0]) == {"id", "section", "markup", "note"}
    first = state.items[0]
    assert (
        call(
            server + "/api/decide", {"id": first["id"], "status": "ok", "markup": first["markup"]}
        )[0]
        == 200
    )
    assert (
        call(server + "/api/decide", {"id": first["id"], "status": "ok", "markup": "başka cümle"})[
            0
        ]
        == 400
    )
    # altın küme uçları bu durumda yok: 404, hata yok
    assert call(server + "/api/clear_after", {"id": first["id"]})[0] == 404
    assert call(server + "/api/reason", {"id": first["id"], "reason": "x"})[0] == 404
    assert call(server + "/api/state", host="evil.example.com")[0] == 403
