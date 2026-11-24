"""ONNX Runtime GenAI modelini aynı test alt kümesinde ölçer (doğruluk + gecikme, CPU).

python -m skymind.optimize.eval_onnx --model-dir outputs/onnx/int4-cpu --name onnx-int4-cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from skymind.eval import Example, evaluate
from skymind.mlops.tracking import log_run
from skymind.optimize.eval_gguf import percentile, pick_rows
from skymind.planner.infer import extract_json
from skymind.planner.prompt import SYSTEM_SHORT, messages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--experiment", default="planner-quant")
    a = ap.parse_args()

    import onnxruntime_genai as og

    rows = pick_rows(a.data, a.limit)
    model: Any = og.Model(str(a.model_dir))
    tok: Any = og.Tokenizer(model)

    def ask(command: str) -> tuple[str, float, int]:
        prompt = tok.apply_chat_template(
            json.dumps(messages(command, system=SYSTEM_SHORT)), add_generation_prompt=True
        )
        ids = tok.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + a.max_tokens, do_sample=False)
        gen = og.Generator(model, params)
        t0 = time.perf_counter()
        gen.append_tokens(ids)
        out: list[int] = []
        while not gen.is_done():
            gen.generate_next_token()
            out.append(int(gen.get_next_tokens()[0]))
        dt = time.perf_counter() - t0
        return str(tok.decode(out)), dt, len(out)

    for r in rows[: a.warmup]:
        ask(r["command"])
    preds: list[str] = []
    lat: list[float] = []
    toks: list[int] = []
    for r in rows:
        text, dt, n = ask(r["command"])
        preds.append(text)
        lat.append(dt)
        toks.append(n)

    strict = evaluate(Example(p, r["gold"]) for p, r in zip(preds, rows, strict=True))
    lenient = evaluate(
        Example(extract_json(p), r["gold"]) for p, r in zip(preds, rows, strict=True)
    )
    size = sum(f.stat().st_size for f in a.model_dir.glob("model.onnx*")) / 1048576
    metrics: dict[str, float] = {
        **strict.metrics,
        **{f"lenient_{k}": v for k, v in lenient.metrics.items()},
        "size_mb": size,
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
        params={"model_dir": str(a.model_dir), "runtime": "onnxruntime-genai", "n": len(rows)},
        metrics=metrics,
        artifacts=[out],
    )
    print(json.dumps({k: round(v, 3) for k, v in metrics.items()}, indent=1))


if __name__ == "__main__":
    main()
