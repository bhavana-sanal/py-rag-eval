from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rag_eval.schema import Chunk, ScoredChunk


class Retriever(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


def top_k(chunks: Sequence[Chunk], scores: NDArray[np.float64], k: int) -> list[ScoredChunk]:
    """Highest-scoring chunks first; ties keep corpus order so runs are reproducible."""
    order = np.argsort(-scores, kind="stable")[:k]
    return [ScoredChunk(chunks[i], float(scores[i])) for i in order]
