"""Altın küme web sunucusu: durum, karar kaydı, geri alma, güvenlik denetimi (gerçek HTTP)."""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from skymind.data import review
from skymind.data.review_web import ReviewState, make_handler
from tests.test_review import make_candidates


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, ReviewState, Path]]:
    src, dec = tmp_path / "c.jsonl", tmp_path / "d.jsonl"
    make_candidates(src)
    state = ReviewState(src, dec)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", state, dec
    httpd.shutdown()
    httpd.server_close()


def call(url: str, body: object | None = None, host: str | None = None) -> tuple[int, dict]:  # type: ignore[type-arg]
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_state_are_served(server: tuple[str, ReviewState, Path]) -> None:
    base, _, _ = server
    with urllib.request.urlopen(base + "/", timeout=5) as r:
        assert r.status == 200 and b"SkyMind" in r.read()
    status, snap = call(base + "/api/state")
    assert status == 200 and len(snap["items"]) == 3 and snap["decisions"] == {}
    assert set(snap["items"][0]) == {"id", "command", "gold", "category"}


def test_decide_persists_to_the_same_file_the_terminal_tool_reads(
    server: tuple[str, ReviewState, Path],
) -> None:
    base, state, dec = server
    first = state.items[0]
    assert (
        call(
            base + "/api/decide", {"id": first["id"], "status": "ok", "command": first["command"]}
        )[0]
        == 200
    )
    second = state.items[1]
    s, out = call(
        base + "/api/decide", {"id": second["id"], "status": "ok", "command": "başka bir komut"}
    )
    assert s == 200 and out["status"] == "edited"  # komut değişti: durumu sunucu belirler
    third = state.items[2]
    call(base + "/api/decide", {"id": third["id"], "status": "bad", "command": third["command"]})
    saved = review.load_decisions(dec)  # terminal aracının okuduğu dosya
    assert {d["status"] for d in saved.values()} == {"ok", "edited", "bad"}
    assert review.stats(dec.with_name("c.jsonl"), dec) == {
        "total": 3, "ok": 1, "edited": 1, "bad": 1, "pending": 0,
    }  # fmt: skip
    # geri al
    assert call(base + "/api/undo", {"id": third["id"]})[0] == 200
    assert third["id"] not in review.load_decisions(dec)


@pytest.mark.parametrize(
    "body",
    [
        {"id": "yok", "status": "ok", "command": "x"},
        {"id": "a", "status": "belki", "command": "x"},
        {"id": "a", "status": "ok", "command": "   "},
        {"id": "a", "status": "ok", "command": "x" * 501},
        {"status": "ok", "command": "x"},
    ],
)
def test_invalid_decisions_are_rejected_and_nothing_is_written(
    server: tuple[str, ReviewState, Path], body: dict[str, object]
) -> None:
    base, _, dec = server
    status, out = call(base + "/api/decide", body)
    assert status == 400 and "error" in out and not dec.exists()


def test_foreign_host_header_is_refused(server: tuple[str, ReviewState, Path]) -> None:
    base, _, _ = server
    assert call(base + "/api/state", host="evil.example.com")[0] == 403
    assert call(base + "/api/undo", {"id": "a"}, host="evil.example.com")[0] == 403
    assert call(base + "/api/state", host="localhost:8765")[0] == 200


def test_reason_can_be_added_to_a_bad_decision_only_and_is_validated(
    server: tuple[str, ReviewState, Path],
) -> None:
    base, state, dec = server
    a, b = state.items[0], state.items[1]
    call(base + "/api/decide", {"id": a["id"], "status": "bad", "command": a["command"]})
    call(base + "/api/decide", {"id": b["id"], "status": "ok", "command": b["command"]})
    assert call(base + "/api/reason", {"id": a["id"], "reason": "gereksiz_soru"})[0] == 200
    assert review.load_decisions(dec)[a["id"]]["reason"] == "gereksiz_soru"
    assert review.reason_counts(dec) == {"gereksiz_soru": 1}
    # aynı kodu tekrar göndermek değil, boş göndermek temizler
    assert call(base + "/api/reason", {"id": a["id"], "reason": ""})[0] == 200
    assert review.reason_counts(dec) == {"belirtilmedi": 1}
    for body in (
        {"id": a["id"], "reason": "uydurma_kod"},  # bilinmeyen neden
        {"id": b["id"], "reason": "gereksiz_soru"},  # doğru işaretlenmiş adaya neden eklenemez
        {"id": "yok", "reason": "gereksiz_soru"},
    ):
        assert call(base + "/api/reason", body)[0] == 400


def test_decide_keeps_reason_only_for_bad_and_snapshot_returns_it(
    server: tuple[str, ReviewState, Path],
) -> None:
    base, state, dec = server
    a, b = state.items[0], state.items[1]
    call(
        base + "/api/decide",
        {"id": a["id"], "status": "bad", "command": a["command"], "reason": "gereksiz_soru"},
    )
    call(
        base + "/api/decide",
        {"id": b["id"], "status": "ok", "command": b["command"], "reason": "gereksiz_soru"},
    )
    saved = review.load_decisions(dec)
    assert saved[a["id"]]["reason"] == "gereksiz_soru" and saved[b["id"]]["reason"] == ""
    snap = call(base + "/api/state")[1]["decisions"]
    assert snap[a["id"]]["reason"] == "gereksiz_soru"
    assert (
        call(
            base + "/api/decide",
            {"id": a["id"], "status": "bad", "command": "x", "reason": "uydurma"},
        )[0]
        == 400
    )


def test_clear_after_removes_only_later_decisions_and_keeps_a_backup(
    server: tuple[str, ReviewState, Path],
) -> None:
    base, state, dec = server
    for it in state.items:  # üçünü de hatalı işaretle (yanlışlıkla toplu hata senaryosu)
        call(base + "/api/decide", {"id": it["id"], "status": "bad", "command": it["command"]})
    first = state.items[0]
    s, out = call(base + "/api/clear_after", {"id": first["id"]})
    assert s == 200 and out["cleared"] == 2 and out["backup"] == "d.bak.jsonl"
    assert set(review.load_decisions(dec)) == {first["id"]}  # öncekiler korunur
    backup = dec.with_name("d.bak.jsonl")
    assert len(review.load_decisions(backup)) == 3  # yedek silmeden önceki hali
    # silinecek karar kalmadıysa yedek yenilenmez ve 0 döner
    backup.unlink()
    assert call(base + "/api/clear_after", {"id": first["id"]})[1]["cleared"] == 0
    assert not backup.exists()
    assert call(base + "/api/clear_after", {"id": "yok"})[0] == 400
