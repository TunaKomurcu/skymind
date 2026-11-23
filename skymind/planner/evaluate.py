"""CLI: planlayıcıyı bir split üzerinde değerlendirir, MLflow'a ve outputs/'a yazar.

python -m skymind.planner.evaluate --name zeroshot --limit 300
python -m skymind.planner.evaluate --name sft-v1 --adapter outputs/sft-v1 --limit 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from skymind.eval import Example, evaluate
from skymind.mlops import log_run
from skymind.planner.infer import DEFAULT_MODEL, extract_json, generate, load
from skymind.planner.prompt import SYSTEM_PROMPT, SYSTEM_SHORT, load_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", type=Path, default=Path("data/processed/test.jsonl"))
    ap.add_argument("--limit", type=int, default=300, help="0 = tümü")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--prompt", choices=["long", "short"], default=None)
    ap.add_argument("--experiment", default="planner-eval")
    a = ap.parse_args()

    rows = load_split(a.data)
    if a.limit:
        # deterministik, eşit aralıklı alt küme
        step = max(1, len(rows) // a.limit)
        rows = rows[::step][: a.limit]
    which = a.prompt or ("short" if a.adapter else "long")
    system = SYSTEM_SHORT if which == "short" else SYSTEM_PROMPT
    tok, model = load(a.model, a.adapter)

    t0 = time.time()
    preds = generate(
        tok, model, [r["command"] for r in rows], a.batch_size, a.max_new_tokens, system
    )
    secs = time.time() - t0

    strict = evaluate(Example(p, r["gold"]) for p, r in zip(preds, rows, strict=True))
    lenient = evaluate(
        Example(extract_json(p), r["gold"]) for p, r in zip(preds, rows, strict=True)
    )
    metrics = {**strict.metrics, **{f"lenient_{k}": v for k, v in lenient.metrics.items()}}
    metrics["sec_per_example"] = secs / len(rows)

    out = Path("outputs") / "eval" / f"{a.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for p, r in zip(preds, rows, strict=True):
            row = {"id": r["id"], "command": r["command"], "gold": r["gold"], "pred": p}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_run(
        a.experiment,
        a.name,
        params={"model": a.model, "adapter": a.adapter, "n": len(rows), "prompt": which},
        metrics=metrics,
        artifacts=[out],
    )
    print(json.dumps({k: round(v, 3) for k, v in metrics.items()}, indent=1))


if __name__ == "__main__":
    main()
