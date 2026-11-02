"""Experiment configuration, loaded from TOML and validated with pydantic.

Unknown keys are rejected everywhere so a typo in a config file fails loudly instead
of silently falling back to a default.
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FixedChunkerConfig(_Strict):
    type: Literal["fixed"]
    size: PositiveInt = 200
    overlap: Annotated[int, Field(ge=0)] = 40

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        return self


class RecursiveChunkerConfig(_Strict):
    type: Literal["recursive"]
    max_words: PositiveInt = 200


class SentenceWindowChunkerConfig(_Strict):
    type: Literal["sentence_window"]
    window: Annotated[int, Field(ge=0)] = 1


ChunkerConfig = Annotated[
    FixedChunkerConfig | RecursiveChunkerConfig | SentenceWindowChunkerConfig,
    Field(discriminator="type"),
]


class _RetrieverBase(_Strict):
    mmr_lambda: Annotated[float, Field(ge=0, le=1)] | None = None
    """Enable MMR diversification with this relevance/diversity trade-off."""
    mmr_fetch_k: PositiveInt = 20


class BM25Config(_RetrieverBase):
    type: Literal["bm25"]
    k1: Annotated[float, Field(ge=0)] = 1.5
    b: Annotated[float, Field(ge=0, le=1)] = 0.75


class DenseConfig(_RetrieverBase):
    type: Literal["dense"]


class HybridConfig(_RetrieverBase):
    type: Literal["hybrid"]
    members: list[str] = Field(min_length=2)
    rrf_k: PositiveInt = 60
    depth: PositiveInt = 50


RetrieverConfig = Annotated[BM25Config | DenseConfig | HybridConfig, Field(discriminator="type")]


class LsaEmbedderConfig(_Strict):
    type: Literal["lsa"] = "lsa"
    dim: PositiveInt = 128


class SentenceTransformerEmbedderConfig(_Strict):
    type: Literal["sentence-transformers"]
    model: str = "all-MiniLM-L6-v2"


EmbedderConfig = Annotated[
    LsaEmbedderConfig | SentenceTransformerEmbedderConfig, Field(discriminator="type")
]


class EndpointConfig(_Strict):
    """An OpenAI-compatible chat endpoint. ``base_url`` falls back to ``$OPENAI_BASE_URL``."""

    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: PositiveInt = 60


class ExtractiveGeneratorConfig(_Strict):
    type: Literal["extractive"] = "extractive"


class LLMGeneratorConfig(EndpointConfig):
    type: Literal["openai"]
    temperature: Annotated[float, Field(ge=0)] = 0.0


GeneratorConfig = Annotated[
    ExtractiveGeneratorConfig | LLMGeneratorConfig, Field(discriminator="type")
]


class JudgeConfig(EndpointConfig):
    cache_dir: Path = Path(".rag-eval-cache")


class GridConfig(_Strict):
    chunker: list[str] = Field(min_length=1)
    retriever: list[str] = Field(min_length=1)
    k: list[PositiveInt] = Field(min_length=1)


class ExperimentConfig(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    corpus: Path
    questions: Path
    chunkers: dict[str, ChunkerConfig]
    retrievers: dict[str, RetrieverConfig]
    grid: GridConfig
    embedder: EmbedderConfig = LsaEmbedderConfig()
    generator: GeneratorConfig = ExtractiveGeneratorConfig()
    judge: JudgeConfig | None = None
    bootstrap_resamples: PositiveInt = 2000
    seed: int = 0

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        for name in self.grid.chunker:
            if name not in self.chunkers:
                raise ValueError(f"grid references unknown chunker {name!r}")
        for name in self.grid.retriever:
            if name not in self.retrievers:
                raise ValueError(f"grid references unknown retriever {name!r}")
        for name, retriever in self.retrievers.items():
            if isinstance(retriever, HybridConfig):
                for member in retriever.members:
                    target = self.retrievers.get(member)
                    if target is None:
                        raise ValueError(f"hybrid {name!r} references unknown {member!r}")
                    if isinstance(target, HybridConfig):
                        raise ValueError(f"hybrid {name!r} cannot nest hybrid {member!r}")
        return self


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One point of the experiment grid."""

    chunker: str
    retriever: str
    k: int

    @property
    def id(self) -> str:
        return f"{self.chunker}/{self.retriever}/k={self.k}"


def expand_grid(config: ExperimentConfig) -> list[RunSpec]:
    """Cartesian product of chunker x retriever x k, in config order."""
    grid = config.grid
    return [RunSpec(*point) for point in itertools.product(grid.chunker, grid.retriever, grid.k)]


def load_experiment(path: Path) -> ExperimentConfig:
    """Parse a TOML experiment file. Relative paths resolve against the file's directory."""
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    config = ExperimentConfig.model_validate(raw)
    base = path.parent
    updates: dict[str, object] = {
        "corpus": base / config.corpus,
        "questions": base / config.questions,
    }
    if config.judge is not None:
        updates["judge"] = config.judge.model_copy(
            update={"cache_dir": base / config.judge.cache_dir}
        )
    return config.model_copy(update=updates)
