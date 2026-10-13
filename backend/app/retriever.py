"""Hybrid retrieval: BM25 (keyword) + dense (semantic), fused with Reciprocal Rank Fusion."""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .ingest import Chunk
from .text import coverage, tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / self.n
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        for i, tf in enumerate(self.tf):
            norm = 1 - self.b + self.b * len(self.docs[i]) / self.avgdl
            for t in q:
                f = tf.get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out


class TfidfDense:
    """Offline stand-in for embeddings so the repo runs without any API key."""

    def __init__(self, docs: list[list[str]]):
        vocab = sorted({t for d in docs for t in d})
        self.index = {t: i for i, t in enumerate(vocab)}
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in vocab])
        self.matrix = np.vstack([self._vec(d) for d in docs])

    def _vec(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.index))
        for t, c in Counter(toks).items():
            if t in self.index:
                v[self.index[t]] = (1 + math.log(c)) * self.idf[self.index[t]]
        n = np.linalg.norm(v)
        return v / n if n else v

    def scores(self, query: str, q: list[str]) -> np.ndarray:
        return self.matrix @ self._vec(q)


class EmbeddingDense:
    """Azure OpenAI embeddings."""

    def __init__(self, texts: list[str], embed_fn):
        self.embed_fn = embed_fn
        m = np.array(embed_fn(texts), dtype=float)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def scores(self, query: str, q: list[str]) -> np.ndarray:
        v = np.array(self.embed_fn([query])[0], dtype=float)
        return self.matrix @ (v / np.linalg.norm(v))


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embed_fn=None, rrf_k: int = 60):
        self.chunks, self.rrf_k = chunks, rrf_k
        toks = [tokens(c.search_text) for c in chunks]
        self.bm25 = BM25(toks)
        self.dense = (EmbeddingDense([c.search_text for c in chunks], embed_fn)
                      if embed_fn else TfidfDense(toks))
        self.dense_kind = "embeddings" if embed_fn else "tfidf"

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = tokens(query)
        b, d = self.bm25.scores(q), self.dense.scores(query, q)
        rb = {int(i): r for r, i in enumerate(np.argsort(-b)) if b[i] > 0}
        rd = {int(i): r for r, i in enumerate(np.argsort(-d)) if d[i] > 0}
        fused = []
        for i in set(rb) | set(rd):
            s = sum(1 / (self.rrf_k + r + 1) for r in (rb.get(i), rd.get(i)) if r is not None)
            # light rerank: reward chunks that contain more of the query's terms
            s += 0.01 * coverage(query, self.chunks[i].search_text)
            fused.append((s, i))
        fused.sort(reverse=True)
        return [Hit(self.chunks[i], round(s, 4), rb.get(i), rd.get(i)) for s, i in fused[:k]]
