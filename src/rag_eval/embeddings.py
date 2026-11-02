"""Text embedders.

:class:`LsaEmbedder` is the default because it is deterministic, dependency-free and
fast enough to refit per experiment run, which keeps tests and CI fully offline.
Neural embeddings are available through the optional ``embeddings`` extra.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from rag_eval.text import tokenize

FloatMatrix = NDArray[np.float64]


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> None:
        """Learn whatever corpus statistics the embedder needs (may be a no-op)."""

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        """Return one row per text."""


def l2_normalize(matrix: FloatMatrix) -> FloatMatrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Texts with no known terms embed to the zero vector; keep them at zero instead of NaN.
    return matrix / np.maximum(norms, 1e-12)


class LsaEmbedder:
    """TF-IDF vectors projected onto their top ``dim`` singular vectors (latent semantic
    analysis). The projection captures term co-occurrence, so queries can match chunks
    that use related but not identical words, unlike BM25.
    """

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: FloatMatrix = np.zeros(0)
        self._components: FloatMatrix = np.zeros((0, 0))

    def fit(self, texts: Sequence[str]) -> None:
        tokenized = [tokenize(t) for t in texts]
        self._vocab = {t: i for i, t in enumerate(sorted({t for ts in tokenized for t in ts}))}
        df = np.zeros(len(self._vocab))
        for tokens in tokenized:
            for term in set(tokens):
                df[self._vocab[term]] += 1
        n = len(texts)
        self._idf = np.log((1 + n) / (1 + df)) + 1
        _, _, vt = np.linalg.svd(self._tfidf(tokenized), full_matrices=False)
        self._components = vt[: self.dim]

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        if not self._vocab:
            raise RuntimeError("LsaEmbedder.embed called before fit")
        projected: FloatMatrix = self._tfidf([tokenize(t) for t in texts]) @ self._components.T
        return l2_normalize(projected)

    def _tfidf(self, tokenized: Sequence[list[str]]) -> FloatMatrix:
        matrix = np.zeros((len(tokenized), len(self._vocab)))
        for row, tokens in enumerate(tokenized):
            for term, count in Counter(tokens).items():
                col = self._vocab.get(term)
                if col is not None:
                    matrix[row, col] = (1 + math.log(count)) * self._idf[col]
        return l2_normalize(matrix)


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` model; install with ``pip install .[embeddings]``."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "the sentence-transformers embedder needs the optional extra: "
                "pip install 'py-rag-eval[embeddings]'"
            ) from exc
        self._model: Any = SentenceTransformer(model)

    def fit(self, texts: Sequence[str]) -> None:
        """Pretrained models need no corpus statistics."""

    def embed(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return l2_normalize(np.asarray(vectors, dtype=np.float64))
