"""Tokenisation and sentence splitting shared by the retrievers, chunkers and generator."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])")
_BLOCK_BREAK = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^#{1,6}\s+")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# fmt: off
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for",
    "from", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "then", "there", "these", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
})
# fmt: on


def s_stem(token: str) -> str:
    """Harman's "S" stemmer: conflates plurals without the aggressiveness of Porter."""
    if len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if not token.endswith(("us", "ss")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and stem plurals."""
    return [s_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split Markdown or plain text into sentences.

    Headings and list items are kept as standalone units; wrapped paragraph lines are
    re-joined first so hard line breaks do not create sentence fragments.
    """
    units: list[str] = []
    for block in _BLOCK_BREAK.split(text):
        paragraph: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _HEADING.match(line) or _LIST_ITEM.match(line):
                if paragraph:
                    units.append(" ".join(paragraph))
                    paragraph = []
                units.append(_LIST_ITEM.sub("", _HEADING.sub("", line)))
            else:
                paragraph.append(line)
        if paragraph:
            units.append(" ".join(paragraph))
    return [s for unit in units for s in _SENTENCE_END.split(unit) if s]
