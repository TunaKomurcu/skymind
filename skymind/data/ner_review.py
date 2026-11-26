"""NER elle sınama setinin (data/gold/ner_manual.txt) insan doğrulaması: web arayüzü + uygulama.

    python -m skymind.data.ner_review --web       # http://127.0.0.1:8766 açılır
    python -m skymind.data.ner_review --stats     # ilerleme
    python -m skymind.data.ner_review --apply     # düzeltmeleri ner_manual.txt'ye uygula (yedekli)

Her cümlede `[metin|ETİKET]` işaretlemesinin doğru olup olmadığına karar verilir; gerekirse işaretleme
düzeltilir (cümle metni değiştirilemez, yalnızca etiket sınırları ve türleri). Kararlar
``ner_manual_reviewed.jsonl`` dosyasına yazılır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from skymind.data.review import load_decisions, save_decisions
from skymind.nlp.datasets import parse_marked

PAGE = Path(__file__).with_name("ner_review_web.html")
MARK = re.compile(r"\[([^\]|]+)\|([A-Z]+)\]")
MAX_MARKUP = 600


def plain(markup: str) -> str:
    return MARK.sub(lambda m: m.group(1), markup)


def load_items(path: Path) -> list[dict[str, str]]:
    """Dosyadaki cümleleri sırayla okur: id (işaretlemenin özeti), bölüm, işaretlenmiş metin."""
    items, section = [], "KOLAY"
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("##"):
            section = raw.strip("# ").strip()
        elif raw.strip() and not raw.startswith("#"):
            items.append(
                {
                    "id": hashlib.sha1(raw.encode()).hexdigest()[:10],
                    "section": section,
                    "markup": raw,
                }
            )
    return items


def load_notes(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class NerState:
    """İş parçacığı güvenli durum: cümleler + kararlar + disk (review_web.make_handler ile uyumlu)."""

    def __init__(self, src: Path, decisions_path: Path, notes_path: Path | None = None) -> None:
        self.items = load_items(src)
        self.by_id = {i["id"]: i for i in self.items}
        self.path = decisions_path
        self.decisions = load_decisions(decisions_path)
        self.notes = load_notes(notes_path or src.with_name("ner_claude_notes.json"))
        self._lock = threading.Lock()

    def _note(self, markup: str) -> str:
        return next((n for key, n in self.notes.items() if key in markup), "")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "items": [i | {"note": self._note(i["markup"])} for i in self.items],
                "decisions": {
                    k: {"status": d["status"], "markup": d["markup"]}
                    for k, d in self.decisions.items()
                },
            }

    def decide(self, body: dict[str, Any]) -> dict[str, Any]:
        cid, status, markup = body.get("id"), body.get("status"), body.get("markup")
        item = self.by_id.get(cid) if isinstance(cid, str) else None
        if item is None:
            raise ValueError("bilinmeyen cümle")
        if status not in ("ok", "bad", "edited"):
            raise ValueError("geçersiz durum")
        if not isinstance(markup, str) or not markup.strip() or len(markup) > MAX_MARKUP:
            raise ValueError("geçersiz işaretleme")
        markup = markup.strip()
        if plain(markup) != plain(item["markup"]):
            raise ValueError("cümle metni değiştirilemez, yalnızca etiketler düzeltilebilir")
        parse_marked(
            markup
        )  # bozuk/bilinmeyen etiket, kelimeye denk gelmeyen aralık için ValueError
        if status != "bad":
            status = "edited" if markup != item["markup"] else "ok"
        with self._lock:
            self.decisions[item["id"]] = {
                "id": item["id"],
                "status": status,
                "markup": markup,
                "section": item["section"],
                "ts": int(time.time()),
            }
            save_decisions(self.path, self.decisions)
        return {"saved": item["id"], "status": status}

    def undo(self, body: dict[str, Any]) -> dict[str, Any]:
        cid = body.get("id")
        if not isinstance(cid, str):
            raise ValueError("geçersiz kimlik")
        with self._lock:
            self.decisions.pop(cid, None)
            save_decisions(self.path, self.decisions)
        return {"undone": cid}


def stats(src: Path, decisions_path: Path) -> dict[str, int]:
    items, dec = load_items(src), load_decisions(decisions_path)
    out = {"total": len(items), "ok": 0, "edited": 0, "bad": 0}
    for d in dec.values():
        if d["status"] in out:
            out[d["status"]] += 1
    out["pending"] = len(items) - out["ok"] - out["edited"] - out["bad"]
    return out


def apply(src: Path, decisions_path: Path) -> dict[str, int]:
    """Düzeltilen satırları yazar, 'hatalı' bırakılanları kümeden çıkarır; önce yedek alır."""
    dec = load_decisions(decisions_path)
    lines, out_lines, edited, removed = src.read_text(encoding="utf-8").splitlines(), [], 0, 0
    for raw in lines:
        if raw.startswith("#") or not raw.strip():
            out_lines.append(raw)
            continue
        d = dec.get(hashlib.sha1(raw.encode()).hexdigest()[:10])
        if d is None or d["status"] == "ok":
            out_lines.append(raw)
        elif d["status"] == "edited":
            out_lines.append(d["markup"])
            edited += 1
        else:  # bad
            removed += 1
    s = stats(src, decisions_path)
    n = sum(1 for ln in out_lines if ln.strip() and not ln.startswith("#"))
    reviewed = s["ok"] + s["edited"] + s["bad"]
    for i, ln in enumerate(out_lines):
        if ln.startswith("# El yazımı mini NER"):
            out_lines[i] = f"# El yazımı mini NER sınama seti ({n} cümle). Biçim: [metin|ETİKET]."
        elif ln.startswith("# Yazar:"):
            out_lines[i] = (
                "# Yazar: Claude (asistan); proje sahibi "
                f"{reviewed}/{s['total']} cümleyi gözden geçirdi ({edited} düzeltme, {removed} çıkarma). "
                "Eğitim verisinden bağımsız ama küçük; yalnızca genelleme göstergesidir."
            )
    if edited or removed:
        src.with_name(f"{src.stem}.bak{src.suffix}").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    src.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {**s, "applied_edits": edited, "removed": removed, "sentences_now": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/gold/ner_manual.txt"))
    ap.add_argument("--decisions", type=Path, default=Path("data/gold/ner_manual_reviewed.jsonl"))
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.web:
        from skymind.data.review_web import serve_state

        serve_state(
            NerState(a.src, a.decisions), PAGE, a.decisions, a.port, open_browser=not a.no_browser
        )
    elif a.apply:
        print(json.dumps(apply(a.src, a.decisions), ensure_ascii=False))
    else:
        print(json.dumps(stats(a.src, a.decisions), ensure_ascii=False))


if __name__ == "__main__":
    main()
