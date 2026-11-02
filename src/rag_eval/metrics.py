"""Per-question metrics. Every function is pure and works on plain ids and strings.

Retrieval metrics take ``ranked``: document ids in rank order, without duplicates.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Collection, Mapping, Sequence


def _require(relevant: Collection[str]) -> None:
    if not relevant:
        raise ValueError("a question needs at least one relevant document")


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 if any relevant document appears in the top ``k``, else 0."""
    _require(relevant)
    return float(any(doc in relevant for doc in ranked[:k]))


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top ``k``."""
    _require(relevant)
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """``1 / rank`` of the first relevant document, 0 if none was retrieved."""
    _require(relevant)
    return next((1.0 / r for r, doc in enumerate(ranked, start=1) if doc in relevant), 0.0)


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Normalised DCG with linear gains: ``DCG = sum_i grade_i / log2(i + 1)``."""
    _require(grades)
    dcg = sum(grades.get(doc, 0) / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg


def citation_precision(cited: Sequence[str | None], relevant: Collection[str]) -> float:
    """Fraction of citation markers that point at a passage from a relevant document.

    ``cited`` holds the source document of each cited passage, or ``None`` for a marker
    that does not match any passage (a fabricated citation). No citations scores 0.
    """
    if not cited:
        return 0.0
    return sum(doc in relevant for doc in cited if doc is not None) / len(cited)


def citation_recall(cited: Sequence[str | None], relevant: Collection[str]) -> float:
    """Fraction of relevant documents cited at least once."""
    _require(relevant)
    return len(set(relevant) & {doc for doc in cited if doc is not None}) / len(relevant)


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation and articles, squash spaces."""
    return " ".join(_ARTICLES.sub(" ", text.lower().translate(_PUNCT)).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall after SQuAD normalisation."""
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)
