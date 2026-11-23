"""Şablon + paraphrase -> train/val/test (sızıntısız) + altın aday seti.

Sızıntı kuralı: bir örneğin grubu
* ``plan``  kategorisinde altın planın kendisi (aynı plan -> aynı split; "eve dön" gibi küçük
  altın uzayları test'e sızmaz),
* diğerlerinde şablon komutunun kimliği (paraphrase'ler orijinaliyle aynı split'te kalır).
Aynı komut metni birden fazla split'te geçmez (öncelik: test > val > train).

    python -m skymind.data.build --n 6000 --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from skymind.data.generator import generate, numbers_in

SPLITS = ("test", "val", "train")
GOLD_SIZE = 300


def group_key(row: dict[str, Any]) -> str:
    if row["category"] == "plan":
        return "plan:" + json.dumps(row["gold"], sort_keys=True)
    return "id:" + str(row["group"])


def split_of(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def load_paraphrases(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d["paraphrases"]
    return out


def expand(rows: list[dict[str, Any]], paraphrases: dict[str, list[str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        base = {**r, "group": r["id"]}
        out.append(base)
        for i, p in enumerate(paraphrases.get(r["id"], [])):
            out.append(
                {
                    **base,
                    "id": f"{r['id']}-p{i}",
                    "command": p,
                    "source": "paraphrase",
                    "numbers": numbers_in(p),
                }
            )
    return out


def assign_splits(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for r in rows:
        buckets[split_of(group_key(r))].append(r)
    seen: set[str] = set()
    for s in SPLITS:  # öncelik sırası: test, val, train
        kept = []
        for r in buckets[s]:
            k = r["command"].lower().strip()
            if k not in seen:
                seen.add(k)
                kept.append(r)
        buckets[s] = kept
    return buckets


def gold_candidates(test: list[dict[str, Any]], size: int = GOLD_SIZE, seed: int = 0) -> list[Any]:
    """Test'ten, yalnızca şablon kaynaklı, kategori oranını koruyan altın aday örnekleri."""
    rng = random.Random(seed)
    pool = [r for r in test if r["source"] == "template"]
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in pool:
        by_cat.setdefault(r["category"], []).append(r)
    out: list[dict[str, Any]] = []
    for _cat, items in sorted(by_cat.items()):
        take = max(1, round(size * len(items) / len(pool)))
        out.extend(rng.sample(items, min(take, len(items))))
    return out[:size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(n: int, seed: int, para_path: Path, out_dir: Path, gold_path: Path) -> dict[str, Any]:
    rows = expand(generate(n, seed), load_paraphrases(para_path))
    buckets = assign_splits(rows)
    for s, items in buckets.items():
        write_jsonl(out_dir / f"{s}.jsonl", items)
    gold = gold_candidates(buckets["test"])
    write_jsonl(gold_path, gold)
    stats = {
        s: {
            "n": len(items),
            "category": dict(Counter(r["category"] for r in items)),
            "source": dict(Counter(r["source"] for r in items)),
        }
        for s, items in buckets.items()
    }
    stats["gold_candidates"] = {
        "n": len(gold),
        "category": dict(Counter(r["category"] for r in gold)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), "utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--paraphrases", type=Path, default=Path("data/synth/paraphrases.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--gold", type=Path, default=Path("data/gold/gold_candidates.jsonl"))
    a = ap.parse_args()
    print(
        json.dumps(build(a.n, a.seed, a.paraphrases, a.out, a.gold), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
