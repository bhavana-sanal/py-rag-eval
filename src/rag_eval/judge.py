"""LLM-as-judge scoring of faithfulness and answer relevance, with an on-disk cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from rag_eval.generation import format_context
from rag_eval.llm import ChatClient, Message
from rag_eval.schema import Chunk

RUBRIC = """\
You grade answers produced by a retrieval-augmented question answering system.
Score each criterion with an integer from 1 to 5.

faithfulness - is every claim in the answer supported by the context?
  5: fully supported; 3: partly supported or contains minor unsupported details;
  1: mostly unsupported by, or contradicts, the context.
relevance - does the answer address the question?
  5: directly and completely answers it; 3: partially answers it;
  1: does not address it. A refusal scores 1 unless the context truly lacks the answer.

Respond with a single JSON object and nothing else:
{"faithfulness": <1-5>, "relevance": <1-5>, "reasoning": "<one short sentence>"}"""


class JudgeError(RuntimeError):
    pass


class JudgeVerdict(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    reasoning: str = ""


def build_judge_messages(question: str, passages: Sequence[Chunk], answer: str) -> list[Message]:
    return [
        {"role": "system", "content": RUBRIC},
        {
            "role": "user",
            "content": (
                f"Context:\n{format_context(passages)}\n\nQuestion: {question}\n\nAnswer: {answer}"
            ),
        },
    ]


def parse_verdict(raw: str) -> JudgeVerdict:
    """Validate the judge's reply, tolerating code fences or prose around the JSON."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        raise JudgeError(f"judge reply contains no JSON object: {raw[:200]!r}")
    try:
        return JudgeVerdict.model_validate_json(raw[start : end + 1])
    except ValidationError as exc:
        raise JudgeError(f"invalid judge verdict: {exc}") from exc


class JudgeCache:
    """One JSON file per request hash, so re-running an experiment never pays twice and
    concurrent runs cannot corrupt each other's entries."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, key: str) -> JudgeVerdict | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return JudgeVerdict.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, verdict: JudgeVerdict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(verdict.model_dump_json())
        os.replace(tmp, self.directory / f"{key}.json")


class LLMJudge:
    def __init__(
        self, client: ChatClient, cache: JudgeCache | None = None, max_attempts: int = 2
    ) -> None:
        self.client = client
        self.cache = cache
        self.max_attempts = max_attempts

    def judge(self, question: str, passages: Sequence[Chunk], answer: str) -> JudgeVerdict:
        messages = build_judge_messages(question, passages, answer)
        key = hashlib.sha256(
            json.dumps({"model": self.client.model, "messages": messages}, sort_keys=True).encode()
        ).hexdigest()
        if self.cache is not None and (cached := self.cache.get(key)) is not None:
            return cached
        error: JudgeError | None = None
        for _ in range(self.max_attempts):
            try:
                verdict = parse_verdict(self.client.complete(messages))
            except JudgeError as exc:
                error = exc
                continue
            if self.cache is not None:
                self.cache.put(key, verdict)
            return verdict
        raise JudgeError(f"no valid verdict after {self.max_attempts} attempts") from error
