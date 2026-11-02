import math

import pytest

from rag_eval.retrievers import BM25Retriever
from rag_eval.schema import Chunk

CHUNKS = [
    Chunk("c0", "d0", "apple banana"),
    Chunk("c1", "d1", "apple apple cherry"),
    Chunk("c2", "d2", "durian"),
]


def make(k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.index(CHUNKS)
    return retriever


def test_scores_match_hand_computation() -> None:
    # N = 3, chunk lengths 2, 3, 1 -> avgdl = 2.
    # "apple": df = 2 -> idf = ln(1 + 1.5 / 2.5) = ln(1.6)
    #   c0: tf 1, len 2 -> 1 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 2/2)) = 1
    #   c1: tf 2, len 3 -> 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 3/2)) = 5 / 4.0625
    scores = make().scores("apple")
    assert scores[0] == pytest.approx(math.log(1.6))
    assert scores[1] == pytest.approx(math.log(1.6) * 5 / 4.0625)
    assert scores[2] == 0.0


def test_rare_terms_weigh_more() -> None:
    # "cherry": df = 1 -> idf = ln(1 + 2.5 / 1.5) = ln(8/3); c1: 2.5 / (1 + 2.0625)
    scores = make().scores("cherry")
    assert scores[1] == pytest.approx(math.log(8 / 3) * 2.5 / 3.0625)


def test_query_terms_are_deduplicated() -> None:
    retriever = make()
    assert retriever.scores("apple apple").tolist() == retriever.scores("apple").tolist()


def test_b_zero_disables_length_normalisation() -> None:
    # With b = 0 every chunk is treated as average length, so c1's extra length is free.
    scores = make(b=0.0).scores("apple")
    assert scores[1] == pytest.approx(math.log(1.6) * 2 * 2.5 / (2 + 1.5))


def test_retrieve_ranks_and_skips_non_matching_chunks() -> None:
    hits = make().retrieve("apple", k=10)
    assert [h.chunk.id for h in hits] == ["c1", "c0"]
    assert make().retrieve("unknown words", k=3) == []


def test_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(k1=-1)
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(b=1.5)
