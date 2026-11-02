"""Answer generation from retrieved passages, with numbered citations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_eval.llm import ChatClient, Message
from rag_eval.schema import Chunk
from rag_eval.text import split_sentences, tokenize

SYSTEM_PROMPT = (
    "Answer the question using only the numbered context passages. "
    "Cite the passage supporting each claim with its number in square brackets, e.g. [2]. "
    "If the passages do not contain the answer, say that you don't know. Be concise."
)

NO_ANSWER = "I don't know based on the provided context."

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[int, ...]
    """1-based passage numbers cited in ``text``, in order of first appearance."""


class Generator(Protocol):
    def generate(self, question: str, passages: Sequence[Chunk]) -> Answer: ...


def format_context(passages: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {p.prompt_text}" for i, p in enumerate(passages, start=1))


def build_messages(question: str, passages: Sequence[Chunk]) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{format_context(passages)}\n\nQuestion: {question}",
        },
    ]


def parse_citations(text: str) -> tuple[int, ...]:
    """Extract ``[n]`` and ``[n, m]`` markers, de-duplicated in order of appearance."""
    numbers = (int(n) for group in _CITATION.findall(text) for n in group.split(","))
    return tuple(dict.fromkeys(numbers))


class ExtractiveGenerator:
    """Deterministic, offline baseline: answers with the single passage sentence that
    shares the most terms with the question, cited to its passage.

    It is a floor to beat, and lets the whole pipeline run in tests without a model.
    """

    def generate(self, question: str, passages: Sequence[Chunk]) -> Answer:
        query_terms = set(tokenize(question))
        best: tuple[int, str, int] | None = None
        for number, passage in enumerate(passages, start=1):
            for sentence in split_sentences(passage.prompt_text):
                overlap = len(query_terms & set(tokenize(sentence)))
                # Strict ">" keeps the earliest (highest-ranked) sentence on ties.
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, sentence, number)
        if best is None:
            return Answer(NO_ANSWER, ())
        _, sentence, number = best
        return Answer(f"{sentence} [{number}]", (number,))


class LLMGenerator:
    def __init__(self, client: ChatClient, temperature: float = 0.0) -> None:
        self.client = client
        self.temperature = temperature

    def generate(self, question: str, passages: Sequence[Chunk]) -> Answer:
        text = self.client.complete(
            build_messages(question, passages), temperature=self.temperature
        )
        return Answer(text, parse_citations(text))
