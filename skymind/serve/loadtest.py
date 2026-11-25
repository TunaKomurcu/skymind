"""Plan servisleri için kapalı döngü yük testi (C++ ve Python servisi aynı betikle ölçülür).

    python -m skymind.serve.loadtest --url http://127.0.0.1:8091 --name cpp-q5km \
        --levels 1,2,4,8 --requests 80

Her eşzamanlılık düzeyinde N işçi, toplam ``--requests`` isteğe ulaşılana dek art arda istek
gönderir. Komutlar test kümesinden deterministik seçilir (iki servis aynı girdiyi görür).
Gecikme istemci tarafında ölçülür (kuyruk bekleme dahil).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from skymind.optimize.eval_gguf import percentile, pick_rows


async def run_level(
    base: str, commands: list[str], concurrency: int, total: int, timeout: float
) -> dict[str, Any]:
    queue: asyncio.Queue[str] = asyncio.Queue()
    for i in range(total):
        queue.put_nowait(commands[i % len(commands)])
    lat: list[float] = []
    new_tokens: list[int] = []
    valid = errors = 0
    types: dict[str, int] = {}

    async with httpx.AsyncClient(
        base_url=base,
        timeout=timeout,
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
    ) as client:

        async def worker() -> None:
            nonlocal valid, errors
            while True:
                try:
                    cmd = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                t0 = time.perf_counter()
                try:
                    r = await client.post("/v1/plan", json={"command": cmd})
                    dt = time.perf_counter() - t0
                    if r.status_code != 200:
                        errors += 1
                        continue
                    body = r.json()
                except (httpx.HTTPError, ValueError):
                    errors += 1
                    continue
                lat.append(dt * 1000)
                new_tokens.append(int(body["usage"]["new_tokens"]))
                valid += bool(body["valid"])
                key = body["type"] or "invalid"
                types[key] = types.get(key, 0) + 1

        t_start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t_start

    ok = len(lat)
    return {
        "concurrency": concurrency,
        "requests": total,
        "ok": ok,
        "errors": errors,
        "wall_s": round(wall, 2),
        "rps": round(ok / wall, 2) if wall else 0.0,
        "tokens_per_s": round(sum(new_tokens) / wall, 1) if wall else 0.0,
        "mean_new_tokens": round(sum(new_tokens) / ok, 1) if ok else 0.0,
        "latency_ms": {
            "mean": round(sum(lat) / ok, 1) if ok else 0.0,
            "p50": round(percentile(lat, 0.50), 1) if ok else 0.0,
            "p95": round(percentile(lat, 0.95), 1) if ok else 0.0,
            "p99": round(percentile(lat, 0.99), 1) if ok else 0.0,
            "max": round(max(lat), 1) if ok else 0.0,
        },
        "valid_rate": round(valid / ok, 3) if ok else 0.0,
        "types": types,
    }


async def amain(a: argparse.Namespace) -> dict[str, Any]:
    rows = pick_rows(a.data, a.pool)
    commands = [r["command"] for r in rows]
    async with httpx.AsyncClient(base_url=a.url, timeout=a.timeout) as c:
        health = (await c.get("/healthz")).json()
        if health.get("status") != "ok":
            raise SystemExit(f"servis hazır değil: {health}")
    if a.warmup:
        await run_level(a.url, commands, 1, a.warmup, a.timeout)
    results = []
    for level in a.levels:
        res = await run_level(a.url, commands, level, a.requests, a.timeout)
        results.append(res)
        lat = res["latency_ms"]
        print(
            f"c={level:<2} rps={res['rps']:<6} tok/s={res['tokens_per_s']:<7} "
            f"p50={lat['p50']:<7} p95={lat['p95']:<7} p99={lat['p99']:<7} "
            f"err={res['errors']} valid={res['valid_rate']}",
            flush=True,
        )
    return {"name": a.name, "url": a.url, "pool": len(commands), "levels": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 4, 8])
    ap.add_argument("--requests", type=int, default=80, help="düzey başına toplam istek")
    ap.add_argument("--pool", type=int, default=200, help="test kümesinden seçilen komut sayısı")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    result = asyncio.run(amain(a))
    out = a.out or Path("outputs/loadtest") / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
