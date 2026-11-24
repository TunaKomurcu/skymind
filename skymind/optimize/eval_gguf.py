"""GGUF varyantlarını llama-server üzerinden aynı test alt kümesinde ölçer.

Doğruluk (eval metrikleri), dosya boyutu ve istek gecikmesi (p50/p95, ardışık istek, batch=1).

    python -m skymind.optimize.eval_gguf --gguf outputs/gguf/skymind-sft-Q4_K_M.gguf \
        --server third_party/llama-cuda/llama-server.exe --name gguf-Q4_K_M --limit 400
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from skymind.eval import Example, evaluate
from skymind.mlops.tracking import log_run
from skymind.planner.infer import extract_json
from skymind.planner.prompt import SYSTEM_SHORT, load_split, messages


def pick_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """planner.evaluate ile aynı deterministik, eşit aralıklı alt küme."""
    rows = load_split(path)
    if limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    return rows


def wait_ready(client: httpx.Client, base: str, proc: subprocess.Popen[bytes], secs: int) -> None:
    deadline = time.time() + secs
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server erken kapandı (kod {proc.returncode})")
        try:
            if client.get(f"{base}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server hazır olmadı")


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--ngl", type=int, default=99, help="GPU'ya yüklenen katman (0 = CPU)")
    ap.add_argument("--threads", type=int, default=0, help="0 = varsayılan")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--experiment", default="planner-quant")
    a = ap.parse_args()

    rows = pick_rows(a.data, a.limit)
    base = f"http://127.0.0.1:{a.port}"
    cmd = [str(a.server), "-m", str(a.gguf), "-ngl", str(a.ngl), "-c", "2048", "-np", "1"]
    cmd += ["--port", str(a.port), "--host", "127.0.0.1", "--jinja", "--no-webui"]
    if a.threads:
        cmd += ["-t", str(a.threads)]
    log_path = Path("outputs") / f"server-{a.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    preds: list[str] = []
    lat: list[float] = []
    toks: list[int] = []
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(timeout=120) as client:
                wait_ready(client, base, proc, 180)

                def ask(command: str) -> tuple[str, float, int]:
                    body = {
                        "messages": messages(command, system=SYSTEM_SHORT),
                        "temperature": 0,
                        "max_tokens": a.max_tokens,
                        "cache_prompt": False,
                    }
                    t0 = time.perf_counter()
                    r = client.post(f"{base}/v1/chat/completions", json=body)
                    dt = time.perf_counter() - t0
                    r.raise_for_status()
                    j = r.json()
                    return (
                        j["choices"][0]["message"]["content"],
                        dt,
                        int(j.get("usage", {}).get("completion_tokens", 0)),
                    )

                for r in rows[: a.warmup]:
                    ask(r["command"])
                for r in rows:
                    text, dt, n = ask(r["command"])
                    preds.append(text)
                    lat.append(dt)
                    toks.append(n)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    strict = evaluate(Example(p, r["gold"]) for p, r in zip(preds, rows, strict=True))
    lenient = evaluate(
        Example(extract_json(p), r["gold"]) for p, r in zip(preds, rows, strict=True)
    )
    metrics: dict[str, float] = {
        **strict.metrics,
        **{f"lenient_{k}": v for k, v in lenient.metrics.items()},
        "size_mb": a.gguf.stat().st_size / 1048576,
        "latency_p50_s": statistics.median(lat),
        "latency_p95_s": percentile(lat, 0.95),
        "latency_mean_s": statistics.fmean(lat),
        "tokens_per_s": sum(toks) / sum(lat),
        "mean_new_tokens": statistics.fmean(toks),
    }
    out = Path("outputs") / "eval" / f"{a.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for p, r in zip(preds, rows, strict=True):
            row = {"id": r["id"], "command": r["command"], "gold": r["gold"], "pred": p}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log_run(
        a.experiment,
        a.name,
        params={"gguf": a.gguf.name, "ngl": a.ngl, "threads": a.threads, "n": len(rows)},
        metrics=metrics,
        artifacts=[out],
    )
    print(json.dumps({k: round(v, 3) for k, v in metrics.items()}, indent=1))


if __name__ == "__main__":
    main()
