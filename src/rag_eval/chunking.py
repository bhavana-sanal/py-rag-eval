"""Chunkers turn a document into retrievable :class:`Chunk` objects.

Every chunker is a small frozen dataclass with a single ``split`` method, so they are
cheap to construct from config and trivially comparable in an experiment grid.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from rag_eval.schema import Chunk, Document
from rag_eval.text import split_sentences


class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...


def chunk_corpus(chunker: Chunker, docs: Iterable[Document]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunker.split(doc)]


def _make_chunks(doc: Document, texts: Iterable[str]) -> list[Chunk]:
    return [Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=t) for i, t in enumerate(texts)]


@dataclass(frozen=True, slots=True)
class FixedSizeChunker:
    """Sliding window of ``size`` words, consecutive windows sharing ``overlap`` words."""

    size: int = 200
    overlap: int = 40

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be in [0, size)")

    def split(self, doc: Document) -> list[Chunk]:
        words = doc.text.split()
        windows: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.size, len(words))
            windows.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - self.overlap
        return _make_chunks(doc, windows)


# Coarsest to finest: Markdown sections, paragraphs, sentences. Below that we cut words.
_SEPARATORS = (
    re.compile(r"\n(?=#{1,6}\s)"),
    re.compile(r"\n\s*\n"),
    re.compile(r"(?<=[.!?])\s+"),
)


def _word_count(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    """Split on the coarsest structural boundary that yields pieces of at most
    ``max_words`` words, then greedily merge neighbouring pieces back up to the limit.

    This keeps sections and paragraphs intact whenever they fit, which tends to give
    retrievers more coherent units than a blind sliding window.
    """

    max_words: int = 200

    def __post_init__(self) -> None:
        if self.max_words <= 0:
            raise ValueError("max_words must be positive")

    def split(self, doc: Document) -> list[Chunk]:
        return _make_chunks(doc, self._split(doc.text.strip(), level=0))

    def _split(self, text: str, level: int) -> list[str]:
        if _word_count(text) <= self.max_words:
            return [text] if text else []
        if level == len(_SEPARATORS):
            words = text.split()
            return [
                " ".join(words[i : i + self.max_words])
                for i in range(0, len(words), self.max_words)
            ]
        parts = [p.strip() for p in _SEPARATORS[level].split(text)]
        pieces = [piece for part in parts if part for piece in self._split(part, level + 1)]
        return self._merge(pieces)

    def _merge(self, pieces: list[str]) -> list[str]:
        merged: list[str] = []
        for piece in pieces:
            if merged and _word_count(merged[-1]) + _word_count(piece) <= self.max_words:
                merged[-1] = f"{merged[-1]}\n\n{piece}"
            else:
                merged.append(piece)
        return merged


@dataclass(frozen=True, slots=True)
class SentenceWindowChunker:
    """Index single sentences but hand the generator ``window`` sentences either side.

    Small units make embeddings and BM25 scores sharp; the window restores enough
    context for the answer to be written.
    """

    window: int = 1

    def __post_init__(self) -> None:
        if self.window < 0:
            raise ValueError("window must be non-negative")

    def split(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.text)
        return [
            Chunk(
                id=f"{doc.id}#{i}",
                doc_id=doc.id,
                text=sentence,
                context=" ".join(sentences[max(0, i - self.window) : i + self.window + 1]),
            )
            for i, sentence in enumerate(sentences)
        ]
