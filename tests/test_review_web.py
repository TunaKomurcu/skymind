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
