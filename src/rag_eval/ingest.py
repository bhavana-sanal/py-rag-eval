"""Loading corpora and labelled question sets from disk."""

from __future__ import annotations

from pathlib import Path

from rag_eval.schema import Document, Question

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


def load_corpus(directory: Path) -> list[Document]:
    """Load every Markdown/text file under ``directory``.

    The document id is the path relative to ``directory`` without its suffix, so
    ``guides/retries.md`` becomes ``guides/retries``.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    paths = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no .md or .txt files found in {directory}")
    return [
        Document(
            id=p.relative_to(directory).with_suffix("").as_posix(),
            text=p.read_text(encoding="utf-8"),
        )
        for p in paths
    ]


def load_questions(path: Path) -> list[Question]:
    """Load a JSONL file with one :class:`Question` object per line."""
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = Question.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: invalid question: {exc}") from exc
        if question.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate question id {question.id!r}")
        seen.add(question.id)
        questions.append(question)
    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions
