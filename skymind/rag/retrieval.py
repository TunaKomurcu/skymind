"""Türkçe BM25, yoğun (dense) arama ve Reciprocal Rank Fusion ile hibrit retrieval.

BM25 tarafı sıfırdan yazıldı (bağımlılık yok): Türkçe'ye uygun küçük harfe çevirme ve isteğe bağlı
"F5" kök bulma (ilk 5 harf; Türkçe IR'da yaygın, basit bir taban çizgisi). Yoğun taraf bir
``Embedder`` arayüzüdür; gerçek model ``HFEmbedder`` (ör. multilingual-e5-small).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

_TOKEN = re.compile(r"[a-zçğıöşüâîû0-9]+(?:-[0-9]+)?")
STOP = {"ve", "bir", "bu", "için", "ile", "mi", "mı", "mu", "mü", "ne", "kaç", "de", "da", "en"}


def turkish_lower(s: str) -> str:
    """İ -> i ve I -> ı (Python'un varsayılan lower()'ı bunu yanlış yapar)."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def tokenize(text: str, stem5: bool = False) -> list[str]:
    toks = _TOKEN.findall(turkish_lower(text))
    toks = [t for t in toks if t not in STOP]
    return [t[:5] for t in toks] if stem5 else toks


class BM25:
    def __init__(self, docs: Sequence[str], stem5: bool = False, k1: float = 1.5, b: float = 0.75):
        self.stem5, self.k1, self.b = stem5, k1, b
        self.tf = [Counter(tokenize(d, stem5)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(1, len(docs))
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query, self.stem5)
        out = []
        for tf, ln in zip(self.tf, self.len, strict=True):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += (
                        self.idf[t]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
                    )
            out.append(s)
        return out


class Embedder(Protocol):
    def embed_queries(self, texts: Sequence[str]) -> Any: ...
    def embed_passages(self, texts: Sequence[str]) -> Any: ...


class HFEmbedder:
    """E5 ailesi için: "query: " / "passage: " önekleri, ortalama havuzlama, L2 normalizasyon."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small", device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tok: Any = AutoTokenizer.from_pretrained(model)
        m: Any = AutoModel.from_pretrained(model)
        m.eval()
        self.model = m.to(device)
        self.device = device

    def _embed(self, texts: Sequence[str], prefix: str, bs: int = 32) -> Any:
        torch = self._torch
        outs = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                enc = self.tok(
                    [prefix + t for t in texts[i : i + bs]],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                h = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                v = (h * mask).sum(1) / mask.sum(1)
                outs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(outs).numpy()

    def embed_queries(self, texts: Sequence[str]) -> Any:
        return self._embed(texts, "query: ")

    def embed_passages(self, texts: Sequence[str]) -> Any:
        return self._embed(texts, "passage: ")


class DenseIndex:
    def __init__(self, embedder: Embedder, docs: Sequence[str]) -> None:
        self.embedder = embedder
        self.mat = embedder.embed_passages(list(docs))

    def scores(self, query: str) -> list[float]:
        q = self.embedder.embed_queries([query])[0]
        return [float(x) for x in self.mat @ q]


def ranking(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def rrf(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None
) -> list[int]:
    """Reciprocal Rank Fusion: skor = sum_i w_i / (k + sıra_i)."""
    ws = list(weights) if weights else [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for r, wi in zip(rankings, ws, strict=True):
        for pos, doc in enumerate(r, start=1):
            fused[doc] = fused.get(doc, 0.0) + wi / (k + pos)
    return sorted(fused, key=lambda d: (-fused[d], d))
