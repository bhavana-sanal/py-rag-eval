"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text.

    ``context`` lets a chunker index a small span (e.g. one sentence) while handing
    the generator a wider window around it.
    """

    id: str
    doc_id: str
    text: str
    context: str | None = None

    @property
    def prompt_text(self) -> str:
        return self.context if self.context is not None else self.text


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class Question(BaseModel):
    """A labelled evaluation question.

    ``relevant`` maps document ids to graded relevance (1 = relevant, higher = more
    relevant). A plain list of ids is accepted and treated as grade 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    relevant: dict[str, Annotated[int, Field(ge=1)]] = Field(min_length=1)
    answer: str | None = None

    @field_validator("relevant", mode="before")
    @classmethod
    def _list_means_binary(cls, value: object) -> object:
        if isinstance(value, list):
            return dict.fromkeys(value, 1)
        return value
