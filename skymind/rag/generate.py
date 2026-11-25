"""Üretken (LLM) RAG yanıtlayıcı + atıf, sadakat ve "cevap yok" davranışı değerlendirmesi.

    python -m skymind.rag.generate --model google/gemma-3-27b-it --n 60

Model, geri getirilen parçalardaki cümleleri ``[parça, cK]`` kimlikleriyle görür ve her iddianın
sonuna atıf koymakla yükümlüdür. Kılavuzda OLMAYAN sorular (olmayan model/özellik) ayrıca sorulur;
doğru davranış ``Bilgi bulunamadı.`` demektir. Ölçüler, otomatik vekil metriklerdir (aşağıda).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skymind.aviation.faithfulness import numbers_in
from skymind.rag.answer import parse_citations, verify
from skymind.rag.corpus import COMPONENTS, QA, Chunk, Corpus, build_corpus, split_sentences
from skymind.rag.retrieval import tokenize, turkish_lower

ABSTAIN = "Bilgi bulunamadı."
SYSTEM = (
    "Sen bir İHA bakım ve operasyon kılavuzu asistanısın. YALNIZCA verilen pasajlardaki bilgiyi "
    "kullan; kendi bilgini ekleme. Her cümlenin sonuna dayandığın pasaj cümlesini "
    "[pasaj-kimliği, cK] biçiminde yaz (örnek: [hata#3, c1]). Soru yalnızca belirli bir araç "
    "modeli için geçerliyse başka modelin değerini verme. Cevap pasajlarda yoksa yalnızca "
    "şunu yaz: "
    f"{ABSTAIN} Kısa ve Türkçe yanıtla."
)


def format_passages(chunks: list[Chunk]) -> str:
    out = []
    for c in chunks:
        sents = "\n".join(f"  c{k}: {s}" for k, s in enumerate(split_sentences(c.text)))
        out.append(f"[{c.id}]\n{sents}")
    return "\n\n".join(out)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    user = f"Pasajlar:\n\n{format_passages(chunks)}\n\nSoru: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def unanswerable_questions(n: int, seed: int = 0) -> list[str]:
    """Kılavuzda olmayan bilgiler: tanımsız model (X4) ve hiç yer almayan özellikler."""
    rng = random.Random(seed)
    qs = [f"X4 modelinde {name} kaç uçuş saatinde bir {verb}?" for name, _alias, verb in COMPONENTS]
    attrs = [
        "ağırlığı kaç kilogramdır",
        "motor gücü kaç watttır",
        "menzili kaç kilometredir",
        "azami yük kapasitesi kaç kilogramdır",
        "gövde malzemesi nedir",
        "pervane çapı kaç inçtir",
        "hangi fabrikada üretilir",
        "kaç kameralıdır",
    ]
    qs += [f"{m} modelinin {a}?" for m in ("X1", "X2", "X3") for a in attrs]
    rng.shuffle(qs)
    return qs[:n]


def _stems(text: str) -> set[str]:
    return set(tokenize(text, stem5=True))


def is_correct(answer: str, gold: str) -> bool:
    """Vekil doğruluk: sayısal cevapta altın sayı(lar) yanıtta geçer; metin cevapta içerik
    kök(ler)inin en az yarısı yanıtta geçer."""
    gnums = numbers_in(gold)
    if gnums:
        return all(g in numbers_in(answer) for g in gnums)
    gs = _stems(gold)
    return bool(gs) and len(gs & _stems(answer)) / len(gs) >= 0.5


def abstained(answer: str) -> bool:
    return re.sub(r"\W+", "", turkish_lower(answer)).startswith("bilgibulunamadı")


def score_answerable(
    qa: QA, answer: str, chunks: list[Chunk], retrieved_ids: list[str]
) -> dict[str, Any]:
    cites = parse_citations(answer)
    valid = verify(cites, chunks)
    by_id = {c.id: c for c in chunks}
    gold_sents = split_sentences(by_id[qa.gold_chunk].text)
    gold_idx = gold_sents.index(qa.gold_sentence) if qa.gold_sentence in gold_sents else -1
    provided = {n for cid in retrieved_ids for n in numbers_in(by_id[cid].text)}
    extra = [n for n in numbers_in(re.sub(r"\[[^\]]*\]", "", answer)) if n not in provided]
    return {
        "retrieved_gold": qa.gold_chunk in retrieved_ids,
        "abstained": abstained(answer),
        "correct": is_correct(answer, qa.answer) and not abstained(answer),
        "n_citations": len(cites),
        "citations_valid": sum(valid),
        "cites_gold_sentence": (qa.gold_chunk, gold_idx) in cites,
        "unsupported_numbers": len(extra),
    }


def evaluate_generation(
    corpus: Corpus,
    rank: Callable[[str], list[int]],
    ask: Callable[[list[dict[str, str]]], str],
    n_answerable: int,
    n_unanswerable: int,
    k: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    rng = random.Random(seed)
    qas = rng.sample(corpus.qas, min(n_answerable, len(corpus.qas)))
    rows: list[dict[str, Any]] = []
    for qa in qas:
        top = [corpus.chunks[i] for i in rank(qa.question)[:k]]
        ans = ask(build_messages(qa.question, top))
        rows.append(
            {"q": qa.question, "kind": qa.kind, "answer": ans}
            | score_answerable(qa, ans, corpus.chunks, [c.id for c in top])
        )
    un: list[dict[str, Any]] = []
    for q in unanswerable_questions(n_unanswerable, seed):
        top = [corpus.chunks[i] for i in rank(q)[:k]]
        ans = ask(build_messages(q, top))
        un.append(
            {
                "q": q,
                "answer": ans,
                "abstained": abstained(ans),
                "has_number": bool(numbers_in(ans)),
            }
        )

    def rate(rs: list[dict[str, Any]], key: str) -> float:
        return round(sum(bool(r[key]) for r in rs) / max(1, len(rs)), 3)

    got = [r for r in rows if r["retrieved_gold"]]
    n_cit = sum(r["n_citations"] for r in rows)
    answered = [r for r in rows if not r["abstained"]]
    un_answered = [r for r in un if not r["abstained"]]
    summary = {
        "n_answerable": len(rows),
        "n_unanswerable": len(un),
        "retrieval_recall_at_k": rate(rows, "retrieved_gold"),
        "correct": rate(rows, "correct"),
        "correct_given_gold_retrieved": rate(got, "correct"),
        "wrongly_abstained": rate(rows, "abstained"),
        "cites_gold_sentence": rate(rows, "cites_gold_sentence"),
        "citation_validity": round(sum(r["citations_valid"] for r in rows) / max(1, n_cit), 3),
        "answered_without_citation": round(
            sum(r["n_citations"] == 0 for r in answered) / max(1, len(answered)), 3
        ),
        "answered_with_unsupported_number": round(
            sum(r["unsupported_numbers"] > 0 for r in answered) / max(1, len(answered)), 3
        ),
        "unanswerable_abstain_rate": rate(un, "abstained"),
        "unanswerable_answered_with_number": rate(un_answered, "has_number"),
    }
    return {"summary": summary, "answerable": rows, "unanswerable": un}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-unanswerable", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/rag-gen.json"))
    a = ap.parse_args()

    from skymind.llm import OpenRouterClient
    from skymind.rag.retrieval import BM25, DenseIndex, HFEmbedder, ranking, rrf

    corpus = build_corpus(a.seed)
    texts = [c.text for c in corpus.chunks]
    bm, dense = BM25(texts, stem5=True), DenseIndex(HFEmbedder(), texts)

    def rank(q: str) -> list[int]:
        return rrf([ranking(bm.scores(q)), ranking(dense.scores(q))])

    client = OpenRouterClient()

    def ask(messages: list[dict[str, str]]) -> str:
        return client.chat(a.model, messages, max_tokens=300, temperature=0.0).text.strip()

    res = evaluate_generation(corpus, rank, ask, a.n, a.n_unanswerable, a.k, a.seed)
    res["model"], res["spent_usd_total"] = a.model, round(client.ledger.spent(), 4)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {a.out}  (toplam harcama: {res['spent_usd_total']} USD)")


if __name__ == "__main__":
    main()
