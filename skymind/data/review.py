"""Altın küme adaylarını terminalde tek tek inceleme aracı (yeniden başlatılabilir).

    python -m skymind.data.review                 # incelemeye başla / kaldığın yerden devam et
    python -m skymind.data.review --stats         # ilerleme özeti
    python -m skymind.data.review --export        # onaylananlardan data/gold/gold.jsonl üret

Her aday için iki şeye bakılır:
  1. KOMUT doğal, anlaşılır bir Türkçe operatör komutu mu?
  2. ALTIN ÇIKTI doğru mu?
     plan          -> komuttaki her sayı/yön/yer doğru yansıtılmış, komutta olmayan bir şey uydurulmamış
     clarification -> gerçekten zorunlu bir bilgi eksik ve soru makul (hedef, irtifa gibi)
     refusal       -> komut gerçekten güvensiz ya da kapsam dışı

Adaylar sabit bir sırayla karıştırılmıştır: ilk N aday her zaman kategorilere dengeli dağılmış bir örnektir,
yani hepsini bitirmeden bırakırsan da (ör. ilk 100) temsilî bir altın küme elde edersin.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from skymind.data.generator import numbers_in
from skymind.schemas import parse_output

DONE = ("ok", "bad", "edited")
HELP = (
    "  [Enter] doğru   [n] hatalı (kümeden at)   [c] komutu düzelt   [g] altın çıktıyı düzelt (JSON)\n"
    "  [k] not ekle    [s] atla (sonra gelir)    [b] geri al          [q] kaydet ve çık   [?] yardım"
)


def load_candidates(path: Path, seed: int = 0) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    random.Random(seed).shuffle(rows)
    return rows


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in rows}


def save_decisions(path: Path, decisions: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    lines = [json.dumps(r, ensure_ascii=False) for r in decisions.values()]
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(path)  # yarım yazılmış dosya bırakma


def render_gold(gold: dict[str, Any]) -> str:
    t = gold.get("type")
    if t == "clarification":
        return (
            f"  SORU: {gold.get('question')}\n  EKSİK ALANLAR: {', '.join(gold.get('missing', []))}"
        )
    if t == "refusal":
        return f"  RET GEREKÇESİ: {gold.get('reason')}"
    lines = []
    for i, s in enumerate(gold.get("steps", []), 1):
        rest = {k: v for k, v in s.items() if k != "action"}
        lines.append(f"  {i}. {s['action']}  {json.dumps(rest, ensure_ascii=False)}")
    return "\n".join(lines) or "  (adımsız plan)"


def record(
    cand: dict[str, Any], status: str, command: str, gold: dict[str, Any], note: str = ""
) -> dict[str, Any]:
    return {
        "id": cand["id"],
        "status": status,
        "command": command,
        "gold": gold,
        "category": cand["category"],
        "note": note,
        "ts": int(time.time()),
    }


def validate_gold(text: str) -> dict[str, Any]:
    """Elle girilen JSON'u şemaya göre doğrular; geçersizse ValueError."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"geçerli JSON değil: {e}") from e
    try:
        parse_output(doc)
    except ValidationError as e:
        first = e.errors()[0]
        raise ValueError(f"şema: {'.'.join(str(p) for p in first['loc'])}: {first['msg']}") from e
    assert isinstance(doc, dict)
    return doc


def export(candidates: Path, decisions_path: Path, out: Path, seed: int = 0) -> dict[str, int]:
    by_id = {c["id"]: c for c in load_candidates(candidates, seed)}
    dec = load_decisions(decisions_path)
    rows = []
    for d in dec.values():
        if d["status"] not in ("ok", "edited") or d["id"] not in by_id:
            continue
        c = by_id[d["id"]]
        rows.append(
            {
                "id": d["id"],
                "command": d["command"],
                "gold": d["gold"],
                "category": d["gold"]["type"],
                "source": "gold-reviewed",
                "numbers": numbers_in(d["command"]),
                "group": c.get("group", d["id"]),
                "rejected": c.get("rejected"),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return stats(candidates, decisions_path, seed) | {"exported": len(rows)}


def stats(candidates: Path, decisions_path: Path, seed: int = 0) -> dict[str, int]:
    cands = load_candidates(candidates, seed)
    dec = load_decisions(decisions_path)
    out = {"total": len(cands), "ok": 0, "edited": 0, "bad": 0}
    for d in dec.values():
        if d["status"] in out:
            out[d["status"]] += 1
    out["pending"] = len(cands) - out["ok"] - out["edited"] - out["bad"]
    return out


def _prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return "q"


def run(candidates: Path, decisions_path: Path, seed: int = 0) -> None:
    cands = load_candidates(candidates, seed)
    dec = load_decisions(decisions_path)
    skipped: set[str] = set()
    history: list[str] = []  # bu oturumdaki karar sırası (geri al için)
    print(__doc__)
    print(HELP)
    while True:
        todo = [c for c in cands if c["id"] not in dec and c["id"] not in skipped]
        if not todo:
            if skipped:
                skipped.clear()
                continue
            print(
                "\nTüm adaylar incelendi. Dışa aktarmak için: python -m skymind.data.review --export"
            )
            return
        cand = todo[0]
        done = len(cands) - len(todo) - len(skipped)
        print(f"\n━━━ {done + 1}/{len(cands)}  [{cand['category']}] ━━━")
        print(f"KOMUT: {cand['command']}")
        print(render_gold(cand["gold"]))
        cmd, gold, note = cand["command"], cand["gold"], ""
        while True:
            key = _prompt("> ").strip().lower()
            if key in ("", "y", "e"):
                status = "edited" if (cmd != cand["command"] or gold != cand["gold"]) else "ok"
                dec[cand["id"]] = record(cand, status, cmd, gold, note)
                history.append(cand["id"])
                break
            if key == "n":
                dec[cand["id"]] = record(cand, "bad", cmd, gold, note or "hatalı")
                history.append(cand["id"])
                break
            if key == "s":
                skipped.add(cand["id"])
                break
            if key == "c":
                new = _prompt("yeni komut: ").strip()
                if new:
                    cmd = new
                    print(f"KOMUT: {cmd}")
                continue
            if key == "g":
                print("güncel altın çıktı JSON'u:\n  " + json.dumps(gold, ensure_ascii=False))
                new = _prompt("yeni JSON (boş = vazgeç): ").strip()
                if new:
                    try:
                        gold = validate_gold(new)
                        print(render_gold(gold))
                    except ValueError as e:
                        print(f"  reddedildi: {e}")
                continue
            if key == "k":
                note = _prompt("not: ").strip()
                continue
            if key == "b":
                if history:
                    last = history.pop()
                    dec.pop(last, None)
                    save_decisions(decisions_path, dec)
                    print("son karar geri alındı; aday yeniden gösterilecek.")
                    break
                print("geri alınacak karar yok.")
                continue
            if key == "q":
                save_decisions(decisions_path, dec)
                print(json.dumps(stats(candidates, decisions_path, seed), ensure_ascii=False))
                return
            print(HELP)
        save_decisions(decisions_path, dec)  # her karardan sonra diske yaz


def main() -> None:
    if sys.platform == "win32":  # Türkçe karakterler için konsolu UTF-8'e al
        import os

        os.system("chcp 65001 >nul")
        for stream in (sys.stdout, sys.stdin):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/gold/gold_candidates.jsonl"))
    ap.add_argument("--decisions", type=Path, default=Path("data/gold/gold_reviewed.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/gold/gold.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.stats:
        print(json.dumps(stats(a.src, a.decisions, a.seed), ensure_ascii=False))
    elif a.export:
        print(json.dumps(export(a.src, a.decisions, a.out, a.seed), ensure_ascii=False))
        print(f"-> {a.out}")
    else:
        run(a.src, a.decisions, a.seed)


if __name__ == "__main__":
    main()
