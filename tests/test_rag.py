"""RAG: korpus bütünlüğü, BM25, RRF, atıf doğrulama (sahte gömücüyle, ağsız)."""

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from skymind.rag.answer import Citation, extractive_answer, parse_citations, verify
from skymind.rag.corpus import MODELS, build_corpus, split_sentences
from skymind.rag.evaluate import evaluate
from skymind.rag.retrieval import BM25, DenseIndex, ranking, rrf, tokenize, turkish_lower

CORPUS = build_corpus(3)


def test_corpus_is_deterministic_and_every_question_has_a_findable_gold() -> None:
    again = build_corpus(3)
    assert [c.text for c in CORPUS.chunks] == [c.text for c in again.chunks]
    assert len(CORPUS.chunks) > 60 and len(CORPUS.qas) > 200
    by_id = {c.id: c for c in CORPUS.chunks}
    for qa in CORPUS.qas:
        assert qa.gold_sentence in by_id[qa.gold_chunk].text
        assert qa.gold_sentence in split_sentences(by_id[qa.gold_chunk].text), qa.question
    kinds = {qa.kind for qa in CORPUS.qas}
    assert kinds == {"lexical", "paraphrase"}


def test_models_share_structure_but_differ_in_values() -> None:
    # aynı olgu üç modelde farklı değer taşımalı, yoksa "model" ayrımı sınanmaz
    by_topic: dict[str, set[str]] = {}
    for qa in CORPUS.qas:
        base = (
            qa.topic.rsplit(":", 1)[0] if qa.topic.startswith(("bakim", "lim_", "bat_")) else None
        )
        if base and qa.kind == "lexical":
            by_topic.setdefault(base, set()).add(qa.answer)
    assert any(len(v) == len(MODELS) for v in by_topic.values())


def test_turkish_lowercase_and_tokenizer() -> None:
    assert turkish_lower("IŞIK İzmir") == "ışık izmir"
    assert tokenize("Pervane kaç saatte bir değiştirilir?") == ["pervane", "saatte", "değiştirilir"]
    assert tokenize("değiştirilir pervaneler", stem5=True) == ["değiş", "perva"]


def test_bm25_ranks_exact_term_first_and_handles_unknown_terms() -> None:
    docs = ["pervane değiştirilir", "batarya şarj edilir", "hata kodu E-117"]
    bm = BM25(docs)
    assert ranking(bm.scores("E-117 hata"))[0] == 2
    assert bm.scores("yok böyle bir kelime") == [0.0, 0.0, 0.0]


def test_rrf_prefers_documents_ranked_well_by_both() -> None:
    assert rrf([[0, 1, 2], [1, 0, 2]])[:2] == [0, 1]
    assert rrf([[3, 1], [1, 3]], weights=[1.0, 5.0])[0] == 1
    assert rrf([[5], [5]]) == [5]


class HashEmbedder:
    """Kelime torbası: her kelime sabit bir rastgele yön. Ağsız, deterministik sahte gömücü."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for t in tokenize(text, stem5=True):
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v += np.random.default_rng(seed).normal(size=self.dim)
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    embed_passages = embed_queries


def test_dense_index_and_full_evaluation_with_fake_embedder() -> None:
    d = DenseIndex(HashEmbedder(), [c.text for c in CORPUS.chunks])
    qa = CORPUS.qas[0]
    best = ranking(d.scores(qa.question))[0]
    assert isinstance(best, int)
    res = evaluate(CORPUS, HashEmbedder())
    assert set(res["methods"]) == {"bm25", "bm25-f5", "dense", "hybrid"}
    for m in res["methods"].values():
        assert 0 <= m["all"]["recall@1"] <= m["all"]["recall@3"] <= m["all"]["recall@5"] <= 1
        assert m["all"]["n"] == len(CORPUS.qas)
    # sözcük örtüşmesine dayalı sahte gömücü bile rastgeleden iyi olmalı (rastgele R@1 ~ 1/91)
    assert res["methods"]["dense"]["lexical"]["recall@1"] > 0.1


def test_extractive_answer_cites_a_real_sentence_of_the_retrieved_chunk() -> None:
    bm = BM25([c.text for c in CORPUS.chunks], stem5=True)
    qa = next(q for q in CORPUS.qas if q.kind == "lexical" and q.topic.startswith("hata"))
    order = ranking(bm.scores(qa.question))[:3]
    cites = extractive_answer(qa.question, CORPUS.chunks, order, max_sentences=2)
    assert 1 <= len(cites) <= 2
    assert all(isinstance(c, Citation) for c in cites)
    assert all(verify([(c.chunk_id, c.sentence_idx)], CORPUS.chunks) == [True] for c in cites)
    assert extractive_answer("x", CORPUS.chunks, []) == []


def test_citation_parsing_and_verification_catch_fabricated_references() -> None:
    chunk_id = CORPUS.chunks[0].id
    n_sent = len(split_sentences(CORPUS.chunks[0].text))
    text = f"Cevap [{chunk_id}, c0] ve [{chunk_id}, c{n_sent + 5}] ve [yok#99, c0]."
    cites = parse_citations(text)
    assert cites == [(chunk_id, 0), (chunk_id, n_sent + 5), ("yok#99", 0)]
    assert verify(cites, CORPUS.chunks) == [True, False, False]


@pytest.mark.parametrize("bad", ["", "atıf yok", "[abc, c1]"])
def test_parse_citations_ignores_malformed(bad: str) -> None:
    assert parse_citations(bad) == []
