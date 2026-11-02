import pytest

from rag_eval.text import s_stem, split_sentences, tokenize


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("queries", "query"),
        ("retries", "retry"),
        ("jobs", "job"),
        ("caches", "cache"),
        ("status", "status"),
        ("class", "class"),
        ("bus", "bus"),
    ],
)
def test_s_stem(word: str, stem: str) -> None:
    assert s_stem(word) == stem


def test_tokenize_lowercases_drops_stopwords_and_stems() -> None:
    assert tokenize("The Workers retry FAILED jobs!") == ["worker", "retry", "failed", "job"]


def test_split_sentences_handles_markdown_structure() -> None:
    text = (
        "# Retries\n\nA job is retried. It waits\nbetween attempts.\n\n- first item\n- second item"
    )
    assert split_sentences(text) == [
        "Retries",
        "A job is retried.",
        "It waits between attempts.",
        "first item",
        "second item",
    ]
