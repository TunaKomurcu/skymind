"""CLI: şablon komutlarını paraphrase eder; yarıda kesilirse kaldığı yerden devam eder.

    python -m skymind.data.run_paraphrase --n 6000 --seed 1 --out data/synth/paraphrases.jsonl

Çıktı satırı: {"id": ..., "command": ..., "paraphrases": [...]} (id = şablon komutunun kimliği).
Bütçe tavanı aşılırsa temiz şekilde durur; ilerleme dosyada kalır.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skymind.data.generator import generate
from skymind.data.paraphrase import paraphrase_all
from skymind.llm import BudgetExceeded, OpenRouterClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("data/synth/paraphrases.jsonl"))
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.exists():
        done = {
            json.loads(line)["id"] for line in args.out.read_text(encoding="utf-8").splitlines()
        }
    todo = [r for r in rows if r["id"] not in done]
    print(f"toplam {len(rows)}, tamamlanmış {len(done)}, kalan {len(todo)}", flush=True)

    client = OpenRouterClient()
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i : i + args.chunk]
        try:
            res = paraphrase_all(client, chunk, k=args.k, workers=args.workers)
        except BudgetExceeded as e:
            print(f"BÜTÇE DURDURDU: {e}", flush=True)
            break
        with args.out.open("a", encoding="utf-8", newline="\n") as f:
            for r in chunk:
                row = {"id": r["id"], "command": r["command"], "paraphrases": res.get(r["id"], [])}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        pct = f"{min(i + args.chunk, len(todo))}/{len(todo)}"
        print(f"{pct}  harcama {client.ledger.spent():.4f} USD", flush=True)


if __name__ == "__main__":
    main()
