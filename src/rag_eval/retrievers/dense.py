from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_eval.embeddings import Embedder, FloatMatrix, l2_normalize
from rag_eval.retrievers.base import top_k
from rag_eval.schema import Chunk, ScoredChunk


class DenseRetriever:
    """Exact cosine similarity against a matrix of chunk embeddings.

    A brute-force matrix product is exact and fast for corpora up to ~10^5 chunks,
    which covers evaluation sets; an ANN index would only add recall noise here.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: FloatMatrix = np.zeros((0, 0))

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        self.embedder.fit(texts)
        self._matrix = l2_normalize(self.embedder.embed(texts))

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        query_vec = l2_normalize(self.embedder.embed([query]))[0]
        return top_k(self._chunks, self._matrix @ query_vec, k)
