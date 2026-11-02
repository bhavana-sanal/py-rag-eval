from collections.abc import Sequence

import numpy as np
import pytest

from rag_eval.embeddings import LsaEmbedder
from rag_eval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MMRReranker,
    RerankedRetriever,
    mmr,
    reciprocal_rank_fusion,
)
from rag_eval.schema import Chunk, ScoredChunk


def ranking(*ids: str) -> list[ScoredChunk]:
    return [ScoredChunk(Chunk(i, i, i), 0.0) for i in ids]


def test_rrf_hand_computed() -> None:
    fused = reciprocal_rank_fusion([ranking("x", "y", "z"), ranking("y", "w")], k=60)
    assert [h.chunk.id for h in fused] == ["y", "x", "w", "z"]
    scores = {h.chunk.id: h.score for h in fused}
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["w"] == pytest.approx(1 / 62)
    assert scores["z"] == pytest.approx(1 / 63)


def test_rrf_breaks_ties_by_first_appearance() -> None:
    fused = reciprocal_rank_fusion([ranking("a"), ranking("b")])
    assert [h.chunk.id for h in fused] == ["a", "b"]


CORPUS = [
    Chunk("pets", "pets", "cats and dogs are popular pets that live with families"),
    Chunk("cats", "cats", "a cat purrs and sleeps most of the day"),
    Chunk("db", "db", "the database stores rows in tables with indexes"),
    Chunk("sql", "sql", "queries select rows from database tables using indexes"),
]


def test_dense_retriever_ranks_by_cosine_similarity() -> None:
    retriever = DenseRetriever(LsaEmbedder(dim=3))
    retriever.index(CORPUS)
    hits = retriever.retrieve("database indexes", k=2)
    assert {h.chunk.id for h in hits} == {"db", "sql"}
    assert hits[0].score >= hits[1].score


def test_hybrid_fuses_members() -> None:
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(LsaEmbedder(dim=3))])
    hybrid.index(CORPUS)
    assert [h.chunk.id for h in hybrid.retrieve("cat purrs", k=1)] == ["cats"]
    with pytest.raises(ValueError, match="two"):
        HybridRetriever([BM25Retriever()])


def test_mmr_trades_relevance_for_diversity() -> None:
    query = np.array([1.0, 0.0])
    candidates = np.array([[1.0, 0.0], [0.99, np.sqrt(1 - 0.99**2)], [0.7, np.sqrt(1 - 0.49)]])
    # Pure relevance keeps the near-duplicate second.
    assert mmr(query, candidates, k=3, lambda_=1.0) == [0, 1, 2]
    # Weighting diversity prefers the dissimilar candidate over the near-duplicate.
    assert mmr(query, candidates, k=2, lambda_=0.3) == [0, 2]


class ReverseReranker:
    def __init__(self) -> None:
        self.seen: list[int] = []
        self.indexed = 0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.indexed = len(chunks)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        self.seen.append(len(candidates))
        return list(reversed(candidates))[:k]


def test_reranker_hook_sees_fetch_k_candidates() -> None:
    reranker = ReverseReranker()
    retriever = RerankedRetriever(BM25Retriever(), reranker, fetch_k=3)
    retriever.index(CORPUS)
    hits = retriever.retrieve("rows tables indexes database", k=1)
    assert reranker.indexed == len(CORPUS)
    assert reranker.seen == [2]  # only two chunks match the query at all
    assert len(hits) == 1


def test_mmr_reranker_returns_at_most_k_distinct_chunks() -> None:
    retriever = RerankedRetriever(BM25Retriever(), MMRReranker(LsaEmbedder(dim=3), 0.5), 10)
    retriever.index(CORPUS)
    hits = retriever.retrieve("cats database", k=2)
    assert len({h.chunk.id for h in hits}) == 2
    assert retriever.retrieve("no overlap at all", k=2) == []
