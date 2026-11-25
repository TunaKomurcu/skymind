"""Cümle bazlı kaynak gösteren çıkarımsal yanıtlayıcı (LLM'siz taban çizgisi) ve atıf doğrulama.

Yanıt, geri getirilen parçalardaki cümleler arasından seçilir; her cümle ``[parça#n, cümle k]``
atfıyla döner. Üretken (LLM) yanıtlayıcı aynı ``Citation`` biçimini kullanır ve ``verify``
ile denetlenir: atıf verilen cümle gerçekten o parçada var mı?
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from skymind.rag.corpus import Chunk, split_sentences
from skymind.rag.retrieval import BM25, ranking


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    sentence_idx: int
    sentence: str

    def __str__(self) -> str:
        return f"[{self.chunk_id}, c{self.sentence_idx}]"


def extractive_answer(
    question: str, chunks: Sequence[Chunk], top_chunks: Sequence[int], max_sentences: int = 1
) -> list[Citation]:
    """Geri getirilen parçalardaki cümleleri soruyla BM25 puanlayıp en iyilerini seçer."""
    cands: list[Citation] = []
    for ci in top_chunks:
        for k, s in enumerate(split_sentences(chunks[ci].text)):
            cands.append(Citation(chunks[ci].id, k, s))
    if not cands:
        return []
    bm = BM25([c.sentence for c in cands], stem5=True)
    order = ranking(bm.scores(question))
    return [cands[i] for i in order[:max_sentences]]


_CITE = re.compile(r"\[([a-z]+#\d+),\s*c(\d+)\]")


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Metindeki ``[doc#n, ck]`` atıflarını (parça kimliği, cümle no) olarak çıkarır."""
    return [(m[1], int(m[2])) for m in _CITE.finditer(text)]


def verify(citations: Sequence[tuple[str, int]], chunks: Sequence[Chunk]) -> list[bool]:
    """Her atıf için: parça var mı ve cümle numarası o parçada geçerli mi?"""
    by_id = {c.id: c for c in chunks}
    out = []
    for cid, k in citations:
        c = by_id.get(cid)
        out.append(c is not None and 0 <= k < len(split_sentences(c.text)))
    return out
