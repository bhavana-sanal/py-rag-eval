from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag_eval.embeddings import Embedder, FloatMatrix, l2_normalize
from rag_eval.retrievers.base import Retriever
from rag_eval.schema import Chunk, ScoredChunk


class Reranker(Protocol):
    """Reorders a candidate list. Plug in a cross-encoder or any custom scorer here."""

    def index(self, chunks: Sequence[Chunk]) -> None:
        """See the corpus once before queries arrive (may be a no-op)."""

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Return at most ``k`` candidates in the new order."""


class RerankedRetriever:
    """Fetches ``fetch_k`` candidates from ``base`` and lets ``reranker`` pick the top k."""

    def __init__(self, base: Retriever, reranker: Reranker, fetch_k: int = 20) -> None:
        self.base = base
        self.reranker = reranker
        self.fetch_k = fetch_k

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.base.index(chunks)
        self.reranker.index(chunks)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        candidates = self.base.retrieve(query, max(k, self.fetch_k))
        return self.reranker.rerank(query, candidates, k)


def mmr(query: FloatMatrix, candidates: FloatMatrix, k: int, lambda_: float) -> list[int]:
    """Maximal Marginal Relevance over unit vectors.

    Greedily picks the candidate maximising
    ``lambda * sim(q, d) - (1 - lambda) * max_{s in selected} sim(d, s)``
    and returns candidate row indices in selection order.
    """
    relevance = candidates @ query
    similarity = candidates @ candidates.T
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        if selected:
            redundancy = similarity[np.ix_(remaining, selected)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        gains = lambda_ * relevance[remaining] - (1 - lambda_) * redundancy
        selected.append(remaining.pop(int(np.argmax(gains))))
    return selected


class MMRReranker:
    """Diversifies results so the context window is not spent on near-duplicate chunks."""

    def __init__(self, embedder: Embedder, lambda_: float = 0.7) -> None:
        if not 0 <= lambda_ <= 1:
            raise ValueError("lambda_ must be in [0, 1]")
        self.embedder = embedder
        self.lambda_ = lambda_
        self._rows: dict[str, int] = {}
        self._matrix: FloatMatrix = np.zeros((0, 0))

    def index(self, chunks: Sequence[Chunk]) -> None:
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self._matrix = l2_normalize(self.embedder.embed(texts))
        self._rows = {c.id: i for i, c in enumerate(chunks)}

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        query_vec = l2_normalize(self.embedder.embed([query]))[0]
        vectors = self._matrix[[self._rows[c.chunk.id] for c in candidates]]
        return [candidates[i] for i in mmr(query_vec, vectors, k, self.lambda_)]
