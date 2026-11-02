from __future__ import annotations

from collections.abc import Sequence

from rag_eval.retrievers.base import Retriever
from rag_eval.schema import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse ranked lists with RRF: ``score(d) = sum_r 1 / (k + rank_r(d))``.

    RRF only looks at ranks, so BM25 scores and cosine similarities never have to be
    put on a common scale. Ties keep first-seen order.
    """
    fused: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(hit.chunk.id, hit.chunk)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [ScoredChunk(chunks[cid], score) for cid, score in ordered]


class HybridRetriever:
    """Runs several retrievers and fuses their top ``depth`` results with RRF."""

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, depth: int = 50) -> None:
        if len(retrievers) < 2:
            raise ValueError("a hybrid retriever needs at least two members")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.depth = depth

    def index(self, chunks: Sequence[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.index(chunks)

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        depth = max(k, self.depth)
        rankings = [r.retrieve(query, depth) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
