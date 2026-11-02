import json
from pathlib import Path

import pytest

from rag_eval.ingest import load_corpus, load_questions


def test_load_corpus_uses_relative_posix_ids(tmp_path: Path) -> None:
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "retries.md").write_text("# Retries")
    (tmp_path / "notes.txt").write_text("plain text")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = load_corpus(tmp_path)
    assert [d.id for d in docs] == ["guides/retries", "notes"]


def test_load_corpus_requires_text_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \.md"):
        load_corpus(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    return path


def test_load_questions_accepts_lists_and_graded_relevance(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "q.jsonl",
        [
            {"id": "q1", "question": "?", "relevant": ["a", "b"], "answer": "x"},
            {"id": "q2", "question": "?", "relevant": {"a": 2}},
        ],
    )
    first, second = load_questions(path)
    assert first.relevant == {"a": 1, "b": 1}
    assert second.relevant == {"a": 2}
    assert second.answer is None


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": "q", "question": "?", "relevant": []}], "invalid question"),
        ([{"id": "q", "question": "?", "relevant": {"a": 0}}], "invalid question"),
        ([{"id": "q", "question": "?", "relevant": ["a"], "typo": 1}], "invalid question"),
        (
            [
                {"id": "q", "question": "?", "relevant": ["a"]},
                {"id": "q", "question": "?", "relevant": ["b"]},
            ],
            "duplicate",
        ),
    ],
)
def test_load_questions_rejects_bad_rows(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_questions(write_jsonl(tmp_path / "q.jsonl", rows))


def test_demo_dataset_is_consistent(demo_dir: Path) -> None:
    doc_ids = {d.id for d in load_corpus(demo_dir / "corpus")}
    questions = load_questions(demo_dir / "questions.jsonl")
    assert len(doc_ids) >= 25
    assert len(questions) >= 40
    assert all(set(q.relevant) <= doc_ids and q.answer for q in questions)
