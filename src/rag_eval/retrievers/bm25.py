from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rag_eval.retrievers.base import top_k
from rag_eval.schema import Chunk, ScoredChunk
from rag_eval.text import tokenize


class BM25Retriever:
    """Okapi BM25 over an in-memory inverted index.

    Uses the Lucene IDF variant ``log(1 + (N - df + 0.5) / (df + 0.5))`` which stays
    positive even for terms that occur in more than half of the chunks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("expected k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._postings: dict[str, tuple[NDArray[np.intp], NDArray[np.float64]]] = {}
        self._idf: dict[str, float] = {}
        self._norm: NDArray[np.float64] = np.zeros(0)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(self._chunks))
        for i, chunk in enumerate(self._chunks):
            counts = Counter(tokenize(chunk.text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append((i, tf))
        n = len(self._chunks)
        avgdl = float(lengths.mean()) if n and lengths.any() else 1.0
        # Length normalisation is per chunk, not per query, so precompute it once.
        self._norm = self.k1 * (1 - self.b + self.b * lengths / avgdl)
        self._postings = {
            term: (np.array([i for i, _ in plist]), np.array([tf for _, tf in plist], float))
            for term, plist in postings.items()
        }
        self._idf = {
            term: math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }

    def scores(self, query: str) -> NDArray[np.float64]:
        scores = np.zeros(len(self._chunks))
        # dict.fromkeys de-duplicates while keeping a fixed summation order.
        for term in dict.fromkeys(tokenize(query)):
            if term not in self._postings:
                continue
            ids, tf = self._postings[term]
            scores[ids] += self._idf[term] * tf * (self.k1 + 1) / (tf + self._norm[ids])
        return scores

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        # Chunks sharing no term with the query are not candidates at all.
        return [hit for hit in top_k(self._chunks, self.scores(query), k) if hit.score > 0]
