import pytest

from rag_eval.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    chunk_corpus,
)
from rag_eval.schema import Document

TEN_WORDS = Document("doc", " ".join(f"w{i}" for i in range(10)))


def words(text: str) -> list[str]:
    return text.split()


def test_fixed_size_windows_and_overlap() -> None:
    chunks = FixedSizeChunker(size=4, overlap=1).split(TEN_WORDS)
    assert [words(c.text) for c in chunks] == [
        ["w0", "w1", "w2", "w3"],
        ["w3", "w4", "w5", "w6"],
        ["w6", "w7", "w8", "w9"],
    ]
    assert [c.id for c in chunks] == ["doc#0", "doc#1", "doc#2"]
    assert all(c.doc_id == "doc" for c in chunks)


def test_fixed_size_last_window_may_be_short_but_never_empty() -> None:
    chunks = FixedSizeChunker(size=4, overlap=2).split(TEN_WORDS)
    assert [words(c.text)[0] for c in chunks] == ["w0", "w2", "w4", "w6"]
    assert words(chunks[-1].text) == ["w6", "w7", "w8", "w9"]
    assert FixedSizeChunker(size=20, overlap=5).split(TEN_WORDS)[0].text == TEN_WORDS.text
    assert FixedSizeChunker().split(Document("empty", "  \n")) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1)])
def test_fixed_size_rejects_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match=r"size|overlap"):
        FixedSizeChunker(size=size, overlap=overlap)


MARKDOWN = Document(
    "guide",
    "# Intro\n\nShort intro paragraph here.\n\n"
    "# Details\n\nFirst detail sentence is here. Second detail sentence follows it.\n\n"
    "Another paragraph with a few more words.",
)


def test_recursive_keeps_document_whole_when_it_fits() -> None:
    chunks = RecursiveChunker(max_words=100).split(MARKDOWN)
    assert len(chunks) == 1
    assert words(chunks[0].text) == words(MARKDOWN.text)


def test_recursive_splits_on_headings_first() -> None:
    chunks = RecursiveChunker(max_words=16).split(MARKDOWN)
    assert chunks[0].text.startswith("# Intro")
    assert chunks[1].text.startswith("# Details")
    assert all(len(words(c.text)) <= 16 for c in chunks)


def test_recursive_falls_back_to_sentences_then_words() -> None:
    chunks = RecursiveChunker(max_words=6).split(MARKDOWN)
    assert all(len(words(c.text)) <= 6 for c in chunks)
    assert "First detail sentence is here." in [c.text for c in chunks]
    # No words are lost or duplicated.
    assert [w for c in chunks for w in words(c.text)] == words(MARKDOWN.text)


def test_recursive_hard_splits_text_without_boundaries() -> None:
    chunks = RecursiveChunker(max_words=3).split(TEN_WORDS)
    assert [len(words(c.text)) for c in chunks] == [3, 3, 3, 1]


def test_sentence_window_indexes_sentences_with_surrounding_context() -> None:
    doc = Document("d", "One. Two. Three. Four.")
    chunks = SentenceWindowChunker(window=1).split(doc)
    assert [c.text for c in chunks] == ["One.", "Two.", "Three.", "Four."]
    assert [c.context for c in chunks] == [
        "One. Two.",
        "One. Two. Three.",
        "Two. Three. Four.",
        "Three. Four.",
    ]
    assert chunks[1].prompt_text == "One. Two. Three."


def test_sentence_window_zero_is_just_the_sentence() -> None:
    chunks = SentenceWindowChunker(window=0).split(Document("d", "One. Two."))
    assert [c.prompt_text for c in chunks] == ["One.", "Two."]


def test_chunk_corpus_concatenates_documents() -> None:
    chunks = chunk_corpus(FixedSizeChunker(size=5, overlap=0), [TEN_WORDS, MARKDOWN])
    assert {c.doc_id for c in chunks} == {"doc", "guide"}
    assert len({c.id for c in chunks}) == len(chunks)
