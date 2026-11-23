"""CLI: ``python -m skymind.data.generate --n 6000 --seed 1 --out data/synth/template.jsonl``."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from skymind.data.generator import generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/synth/template.jsonl"))
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} örnek -> {args.out}  {dict(Counter(r['category'] for r in rows))}")


if __name__ == "__main__":
    main()
