"""Klasik taban çizgileri: TF-IDF + lojistik regresyon (niyet), kural tabanlı etiketleyici (NER).

python -m skymind.nlp.baselines
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from skymind.nlp.datasets import intent_data, load, ner_data
from skymind.nlp.metrics import entity_f1
from skymind.nlp.ner_data import INTENTS, align_example, rule_tagger


def intent_baseline() -> dict[str, Any]:
    xtr, ytr = intent_data("train")
    xte, yte = intent_data("test")
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, C=10.0),
    )
    model.fit(xtr, ytr)
    t0 = time.perf_counter()
    pred = model.predict(xte)
    ms = (time.perf_counter() - t0) / len(xte) * 1000
    return {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "ms_per_example_cpu": round(ms, 3),
        "classes": INTENTS,
        "n_test": len(xte),
    }


def ner_baseline() -> dict[str, Any]:
    """Altın plan kullanmayan kural tabanlı etiketleyici, hizalanabilen test örneklerinde."""
    gold_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []
    for r in load("test"):
        if r["gold"]["type"] != "plan":
            continue
        al = align_example(r["command"], r["gold"])
        if al is None:
            continue
        gold_tags.append(al[1])
        pred_tags.append(rule_tagger(r["command"]))
    res = entity_f1(gold_tags, pred_tags)
    res["n_test"] = {"sentences": len(gold_tags)}
    return res


def main() -> None:
    out = {"intent_tfidf_lr": intent_baseline(), "ner_rules": ner_baseline()}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = Path("outputs/eval/nlp-baselines.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_tr = len(ner_data("train")[0])
    print(f"NER eğitim cümlesi (hizalanmış): {n_tr}")


if __name__ == "__main__":
    main()
