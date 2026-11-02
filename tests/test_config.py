from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval.config import ExperimentConfig, RunSpec, expand_grid, load_experiment

BASE: dict[str, object] = {
    "name": "t",
    "corpus": "corpus",
    "questions": "q.jsonl",
    "chunkers": {
        "fixed": {"type": "fixed", "size": 50, "overlap": 10},
        "sent": {"type": "sentence_window"},
    },
    "retrievers": {
        "bm25": {"type": "bm25"},
        "dense": {"type": "dense"},
        "hybrid": {"type": "hybrid", "members": ["bm25", "dense"]},
    },
    "grid": {"chunker": ["fixed", "sent"], "retriever": ["bm25", "hybrid"], "k": [1, 5, 10]},
}


def config(**overrides: object) -> ExperimentConfig:
    return ExperimentConfig.model_validate(BASE | overrides)


BM25_ONLY_GRID = {"chunker": ["fixed"], "retriever": ["bm25"], "k": [1]}


def test_expand_grid_is_the_cartesian_product_in_config_order() -> None:
    specs = expand_grid(config())
    assert len(specs) == 2 * 2 * 3
    assert specs[:4] == [
        RunSpec("fixed", "bm25", 1),
        RunSpec("fixed", "bm25", 5),
        RunSpec("fixed", "bm25", 10),
        RunSpec("fixed", "hybrid", 1),
    ]
    assert specs[-1].id == "sent/hybrid/k=10"
    assert len({s.id for s in specs}) == len(specs)


def test_defaults() -> None:
    cfg = config()
    assert cfg.generator.type == "extractive"
    assert cfg.embedder.type == "lsa"
    assert cfg.judge is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grid": {"chunker": ["nope"], "retriever": ["bm25"], "k": [1]}}, "unknown chunker"),
        ({"grid": {"chunker": ["fixed"], "retriever": ["nope"], "k": [1]}}, "unknown retriever"),
        ({"grid": {"chunker": ["fixed"], "retriever": ["bm25"], "k": [0]}}, "greater than 0"),
        ({"unexpected": 1}, "Extra inputs"),
        ({"chunkers": {"fixed": {"type": "fixed", "size": 10, "overlap": 10}}}, "overlap"),
        ({"chunkers": {"fixed": {"type": "magic"}}}, "does not match any of the expected tags"),
        (
            {
                "grid": BM25_ONLY_GRID,
                "retrievers": {
                    "bm25": {"type": "bm25"},
                    "h": {"type": "hybrid", "members": ["bm25", "x"]},
                },
            },
            "unknown 'x'",
        ),
        (
            {
                "grid": BM25_ONLY_GRID,
                "retrievers": {
                    "bm25": {"type": "bm25"},
                    "h1": {"type": "hybrid", "members": ["bm25", "h2"]},
                    "h2": {"type": "hybrid", "members": ["bm25", "bm25"]},
                },
            },
            "cannot nest",
        ),
    ],
)
def test_invalid_configs_fail_loudly(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        config(**overrides)


def test_load_experiment_resolves_paths_relative_to_the_file(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    path = exp_dir / "exp.toml"
    path.write_text(
        """
name = "exp"
corpus = "../data/corpus"
questions = "../data/q.jsonl"

[grid]
chunker = ["c"]
retriever = ["r"]
k = [3]

[chunkers.c]
type = "recursive"

[retrievers.r]
type = "bm25"
mmr_lambda = 0.5

[judge]
model = "gpt-5-mini"
"""
    )
    cfg = load_experiment(path)
    assert cfg.corpus == exp_dir / "../data/corpus"
    assert cfg.questions == exp_dir / "../data/q.jsonl"
    assert cfg.judge is not None
    assert cfg.judge.cache_dir == exp_dir / ".rag-eval-cache"


def test_example_experiment_is_valid(example_config: Path) -> None:
    cfg = load_experiment(example_config)
    assert cfg.corpus.is_dir()
    assert cfg.questions.is_file()
    assert len(expand_grid(cfg)) == 24
