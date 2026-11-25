"""Mevcut paraphrases.jsonl'daki sözlük-metni bozulmasını onarır ve yeniden doğrular.

    python -m skymind.data.repair [--path data/synth/paraphrases.jsonl]

Onarılan paraphrase orijinaliyle yeniden ``check_paraphrase`` süzgecinden geçer; geçemeyen atılır.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skymind.data.paraphrase import check_paraphrase, item_text


def repair(path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    stats = {"rows": len(rows), "fixed": 0, "dropped_after_fix": 0, "duplicates": 0}
    for r in rows:
        out: list[str] = []
        for p in r["paraphrases"]:
            t = item_text(p)
            if t != p:
                stats["fixed"] += 1
                if check_paraphrase(r["command"], t) is not None:
                    stats["dropped_after_fix"] += 1
                    continue
            if t in out:
                stats["duplicates"] += 1
                continue
            out.append(t)
        r["paraphrases"] = out
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=Path("data/synth/paraphrases.jsonl"))
    print(json.dumps(repair(ap.parse_args().path)))


if __name__ == "__main__":
    main()
