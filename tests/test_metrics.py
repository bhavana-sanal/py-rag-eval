import math

import pytest

from rag_eval.metrics import (
    citation_precision,
    citation_recall,
    exact_match,
    hit_at_k,
    ndcg_at_k,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)


def test_hit_at_k_respects_cutoff() -> None:
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, {"c"}, 2) == 0.0
    assert hit_at_k(ranked, {"c"}, 3) == 1.0


def test_recall_at_k_counts_relevant_found_in_top_k() -> None:
    # Relevant {b, d, x}; top-3 is a, b, c -> only b found -> 1/3.
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d", "x"}, 3) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "b", "c", "d"], {"b", "d", "x"}, 4) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_relevant() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == pytest.approx(1 / 2)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_hand_computed() -> None:
    grades = {"b": 2, "c": 1, "d": 1}
    # DCG  = 0/log2(2) + 2/log2(3) + 1/log2(4)
    # IDCG = 2/log2(2) + 1/log2(3) + 1/log2(4)
    expected = (2 / math.log2(3) + 1 / 2) / (2 + 1 / math.log2(3) + 1 / 2)
    assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(expected)
    assert expected == pytest.approx(0.56273, abs=1e-5)


def test_ndcg_is_one_for_ideal_ranking_and_zero_for_misses() -> None:
    assert ndcg_at_k(["b", "c", "d"], {"b": 2, "c": 1, "d": 1}, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"b": 1}, 2) == 0.0


def test_retrieval_metrics_reject_unlabelled_questions() -> None:
    with pytest.raises(ValueError, match="relevant"):
        recall_at_k(["a"], set(), 1)


def test_citation_precision_counts_fabricated_markers_as_wrong() -> None:
    # Cited passages came from a (relevant), b (not relevant) and a marker with no passage.
    assert citation_precision(["a", "b", None], {"a"}) == pytest.approx(1 / 3)
    assert citation_precision([], {"a"}) == 0.0


def test_citation_recall_counts_each_relevant_document_once() -> None:
    assert citation_recall(["a", "a", None], {"a", "b"}) == pytest.approx(0.5)


def test_normalize_answer_follows_squad_rules() -> None:
    assert normalize_answer("  The Quick, brown FOX! ") == "quick brown fox"


def test_exact_match_ignores_case_articles_and_punctuation() -> None:
    assert exact_match("The Rust!", "rust") == 1.0
    assert exact_match("Rust 2", "rust") == 0.0


def test_token_f1_hand_computed() -> None:
    # Normalised: "cat sat on mat" vs "cat sat" -> common 2, P = 2/4, R = 2/2.
    assert token_f1("The cat sat on the mat", "a cat sat") == pytest.approx(2 / 3)
    assert token_f1("dog", "cat") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("the", "cat") == 0.0
