"""Altın küme incelemesi için yerel web arayüzü (tek tuşla doğru/hatalı).

    python -m skymind.data.review --web          # http://127.0.0.1:8765 açılır

Sunucu yalnızca 127.0.0.1'e bağlanır; kararlar her tıklamada ``gold_reviewed.jsonl`` dosyasına
yazılır (terminal aracıyla aynı dosya ve biçim, ikisi birbirinin yerine kullanılabilir).
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from skymind.data.review import load_candidates, load_decisions, record, save_decisions

PAGE = Path(__file__).with_name("review_web.html")
MAX_BODY = 20_000
MAX_COMMAND = 500
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


class ReviewState:
    """Sunucudan bağımsız, iş parçacığı güvenli durum: adaylar + kararlar + disk."""

    def __init__(self, src: Path, decisions_path: Path, seed: int = 0) -> None:
        self.items = load_candidates(src, seed)
        self.by_id = {c["id"]: c for c in self.items}
        self.path = decisions_path
        self.decisions = load_decisions(decisions_path)
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "items": [
                    {k: c[k] for k in ("id", "command", "gold", "category")} for c in self.items
                ],
                "decisions": {
                    i: {"status": d["status"], "command": d["command"]}
                    for i, d in self.decisions.items()
                },
            }

    def decide(self, body: dict[str, Any]) -> dict[str, Any]:
        cid, status, command = body.get("id"), body.get("status"), body.get("command")
        cand = self.by_id.get(cid) if isinstance(cid, str) else None
        if cand is None:
            raise ValueError("bilinmeyen aday")
        if status not in ("ok", "bad", "edited"):
            raise ValueError("geçersiz durum")
        if not isinstance(command, str) or not command.strip() or len(command) > MAX_COMMAND:
            raise ValueError("geçersiz komut")
        command = command.strip()
        if status != "bad":  # durumu sunucu belirler: komut değiştiyse edited
            status = "edited" if command != cand["command"] else "ok"
        with self._lock:
            self.decisions[cand["id"]] = record(cand, status, command, cand["gold"])
            save_decisions(self.path, self.decisions)
        return {"saved": cand["id"], "status": status}

    def undo(self, body: dict[str, Any]) -> dict[str, Any]:
        cid = body.get("id")
        if not isinstance(cid, str):
            raise ValueError("geçersiz kimlik")
        with self._lock:
            self.decisions.pop(cid, None)
            save_decisions(self.path, self.decisions)
        return {"undone": cid}


def make_handler(state: ReviewState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # sessiz
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in ALLOWED_HOSTS  # DNS yeniden bağlama saldırısına karşı

        def _send(self, status: int, payload: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, obj: Any) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "host"})
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self._json(200, state.snapshot())
            self._json(HTTPStatus.NOT_FOUND, {"error": "yok"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "host"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "gövde boyutu"})
            try:
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict):
                    raise ValueError("nesne bekleniyordu")
                if self.path == "/api/decide":
                    return self._json(200, state.decide(body))
                if self.path == "/api/undo":
                    return self._json(200, state.undo(body))
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            self._json(HTTPStatus.NOT_FOUND, {"error": "yok"})

    return Handler


def serve(
    src: Path, decisions_path: Path, seed: int = 0, port: int = 8765, open_browser: bool = True
) -> None:
    state = ReviewState(src, decisions_path, seed)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    url = f"http://127.0.0.1:{port}"
    print(f"İnceleme arayüzü: {url}  (kapatmak için Ctrl+C)")
    print(f"Kararlar buraya yazılır: {decisions_path}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatıldı.")
    finally:
        httpd.server_close()
