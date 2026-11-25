"""İki servisin aynı komutlara verdiği ham çıktıları karşılaştırır (aynı model ve istem beklenir).

    python -m skymind.serve.parity --a http://127.0.0.1:8091 --b http://127.0.0.1:8094 --first 60

Komut havuzu yük testiyle aynıdır (``--pool 200``); yük testi her düzeyde havuzun ilk komutlarını
kullandığı için ``--first`` o komutları kapsar.

Açgözlü örneklemede aynı GGUF ve aynı llama.cpp ikilisi aynı metni üretmelidir; farklı çıkarsa
yük testi karşılaştırması aynı işi ölçmüyor demektir. Küçük sayısal farklar (farklı çekirdek/yığın)
nadiren ayrışmaya yol açabilir; bu araç yalnızca sayar ve örnek gösterir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from skymind.eval.sampling import pick_rows


def ask(client: httpx.Client, command: str) -> dict[str, Any]:
    r = client.post("/v1/plan", json={"command": command})
    r.raise_for_status()
    out: dict[str, Any] = r.json()
    return out


def compare(
    a: httpx.Client, b: httpx.Client, commands: list[str]
) -> tuple[int, list[dict[str, Any]]]:
    """(birebir aynı çıktı sayısı, farklı olanların kayıtları)."""
    same, diffs = 0, []
    for cmd in commands:
        ra, rb = ask(a, cmd), ask(b, cmd)
        if ra["raw"] == rb["raw"]:
            same += 1
        else:
            diffs.append(
                {
                    "command": cmd,
                    "a": ra["raw"],
                    "b": rb["raw"],
                    "a_valid": ra["valid"],
                    "b_valid": rb["valid"],
                }
            )
    return same, diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--first", type=int, default=60, help="havuzun ilk kaç komutu")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/loadtest/parity.json"))
    args = ap.parse_args()
    commands = [r["command"] for r in pick_rows(args.data, args.pool)][: args.first]
    with (
        httpx.Client(base_url=args.a, timeout=args.timeout) as ca,
        httpx.Client(base_url=args.b, timeout=args.timeout) as cb,
    ):
        same, diffs = compare(ca, cb, commands)
    res = {"n": len(commands), "identical": same, "different": diffs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"birebir aynı: {same}/{len(commands)}; farklı: {len(diffs)} -> {args.out}")
    for d in diffs[:3]:
        print(f"  {d['command'][:70]}\n   A: {d['a'][:110]}\n   B: {d['b'][:110]}")


if __name__ == "__main__":
    main()
