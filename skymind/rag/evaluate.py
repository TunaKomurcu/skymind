"""Retrieval ve cümle bazlı atıf değerlendirmesi.

    python -m skymind.rag.evaluate --out outputs/eval/rag.json

Yöntemler: bm25 (ham), bm25-f5 (ilk 5 harf kökü), dense (E5), hybrid (RRF: bm25-f5 + dense).
Metrikler soru türüne göre: recall@1/3/5 ve MRR (altın parça), ayrıca ilk 3 parçadan seçilen
cümlenin altın cümle olup olmadığı (``sentence_hit``).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from skymind.rag.answer import extractive_answer
from skymind.rag.corpus import Corpus, build_corpus
from skymind.rag.retrieval import BM25, DenseIndex, Embedder, ranking, rrf

Ranker = Callable[[str], list[int]]


def build_rankers(corpus: Corpus, embedder: Embedder | None) -> dict[str, Ranker]:
    texts = [c.text for c in corpus.chunks]
    bm_raw, bm_f5 = BM25(texts), BM25(texts, stem5=True)
    rankers: dict[str, Ranker] = {
        "bm25": lambda q: ranking(bm_raw.scores(q)),
        "bm25-f5": lambda q: ranking(bm_f5.scores(q)),
    }
    if embedder is not None:
        dense = DenseIndex(embedder, texts)
        rankers["dense"] = lambda q: ranking(dense.scores(q))
        rankers["hybrid"] = lambda q: rrf([ranking(bm_f5.scores(q)), ranking(dense.scores(q))])
    return rankers


def _metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    n = len(ranks)

    def r_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n

    return {
        "n": n,
        "recall@1": round(r_at(1), 3),
        "recall@3": round(r_at(3), 3),
        "recall@5": round(r_at(5), 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def evaluate(corpus: Corpus, embedder: Embedder | None) -> dict[str, Any]:
    ids = [c.id for c in corpus.chunks]
    rankers = build_rankers(corpus, embedder)
    out: dict[str, Any] = {"chunks": len(ids), "questions": len(corpus.qas), "methods": {}}
    for name, rk in rankers.items():
        by_kind: dict[str, list[int | None]] = {"all": [], "lexical": [], "paraphrase": []}
        hits: list[bool] = []
        for qa in corpus.qas:
            order = rk(qa.question)
            rank = next((p for p, i in enumerate(order, 1) if ids[i] == qa.gold_chunk), None)
            by_kind["all"].append(rank)
            by_kind[qa.kind].append(rank)
            cites = extractive_answer(qa.question, corpus.chunks, order[:3])
            hits.append(bool(cites) and cites[0].sentence == qa.gold_sentence)
        out["methods"][name] = {k: _metrics(v) for k, v in by_kind.items()}
        out["methods"][name]["sentence_hit"] = round(sum(hits) / len(hits), 3)
    return out


def failures(corpus: Corpus, rk: Ranker, limit: int = 5) -> list[dict[str, str]]:
    ids = [c.id for c in corpus.chunks]
    bad: list[dict[str, str]] = []
    for qa in corpus.qas:
        order = rk(qa.question)
        if ids[order[0]] != qa.gold_chunk:
            bad.append(
                {"q": qa.question, "gold": qa.gold_chunk, "got": ids[order[0]], "kind": qa.kind}
            )
    return bad[:limit]


def _print(res: dict[str, Any]) -> None:
    print(f"{res['chunks']} parça, {res['questions']} soru")
    for kind in ("all", "lexical", "paraphrase"):
        print(f"\n[{kind}]  yöntem        R@1   R@3   R@5   MRR")
        for name, m in res["methods"].items():
            x = m[kind]
            row = (x["recall@1"], x["recall@3"], x["recall@5"], x["mrr"])
            print(f"        {name:<10} " + " ".join(f"{v:.3f}" for v in row))
    print("\ncümle isabeti (ilk 3 parçadan seçilen cümle = altın cümle):")
    for name, m in res["methods"].items():
        print(f"  {name:<10} {m['sentence_hit']:.3f}")


def aggregate(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Tohumlar arası ortalama ve örneklem standart sapması (n-1)."""
    import statistics

    def ms(xs: list[float]) -> dict[str, float]:
        return {
            "mean": round(statistics.fmean(xs), 3),
            "sd": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
        }

    out: dict[str, Any] = {"seeds": len(runs), "methods": {}}
    for name in runs[0]["methods"]:
        m: dict[str, Any] = {}
        for kind in ("all", "lexical", "paraphrase"):
            m[kind] = {
                k: ms([r["methods"][name][kind][k] for r in runs])
                for k in ("recall@1", "recall@3", "recall@5", "mrr")
            }
        m["sentence_hit"] = ms([r["methods"][name]["sentence_hit"] for r in runs])
        out["methods"][name] = m
    return out


def _print_agg(agg: dict[str, Any]) -> None:
    print(f"\n{agg['seeds']} korpus tohumu, ortalama ± sd")
    for kind in ("all", "lexical", "paraphrase"):
        print(f"\n[{kind}]  yöntem       R@1            R@5            MRR")
        for name, m in agg["methods"].items():
            x = m[kind]
            cells = [
                f"{x[k]['mean']:.3f}±{x[k]['sd']:.3f}" for k in ("recall@1", "recall@5", "mrr")
            ]
            print(f"        {name:<10} " + "  ".join(cells))
    print("\ncümle isabeti:")
    for name, m in agg["methods"].items():
        print(f"  {name:<10} {m['sentence_hit']['mean']:.3f}±{m['sentence_hit']['sd']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--seeds", type=lambda x: [int(i) for i in x.split(",")], default=[3])
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/rag.json"))
    a = ap.parse_args()

    from skymind.rag.retrieval import HFEmbedder

    embedder = None if a.no_dense else HFEmbedder(a.model)  # bir kez yüklenir
    runs = []
    for seed in a.seeds:
        res = evaluate(build_corpus(seed), embedder)
        res["seed"] = seed
        runs.append(res)
    result: dict[str, Any] = {"embedder": None if a.no_dense else a.model, "runs": runs}
    if len(runs) == 1:
        _print(runs[0])
    else:
        result["aggregate"] = aggregate(runs)
        _print_agg(result["aggregate"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
