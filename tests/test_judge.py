from pathlib import Path

import httpx
import pytest

from rag_eval.judge import JudgeCache, JudgeError, LLMJudge, parse_verdict
from rag_eval.llm import ChatClient
from rag_eval.schema import Chunk

PASSAGES = [Chunk("a#0", "a", "The API listens on port 7420.")]
VALID = '{"faithfulness": 5, "relevance": 4, "reasoning": "Supported by [1]."}'


def test_parse_verdict_accepts_fenced_json_with_prose() -> None:
    verdict = parse_verdict(f"Here is my grade:\n```json\n{VALID}\n```")
    assert (verdict.faithfulness, verdict.relevance) == (5, 4)


@pytest.mark.parametrize(
    "raw",
    [
        "I think it is fine.",
        '{"faithfulness": 6, "relevance": 4}',
        '{"faithfulness": 5}',
        '{"faithfulness": "high", "relevance": 4}',
    ],
)
def test_parse_verdict_rejects_invalid_replies(raw: str) -> None:
    with pytest.raises(JudgeError):
        parse_verdict(raw)


def make_judge(replies: list[str], cache_dir: Path | None) -> tuple[LLMJudge, list[int]]:
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        content = replies[min(calls[0], len(replies) - 1)]
        calls[0] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = ChatClient(
        "judge-model",
        base_url="http://judge.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    cache = JudgeCache(cache_dir) if cache_dir else None
    return LLMJudge(client, cache), calls


def test_cache_avoids_repeat_calls_across_instances(tmp_path: Path) -> None:
    judge, calls = make_judge([VALID], tmp_path)
    first = judge.judge("Which port?", PASSAGES, "7420 [1]")
    second = judge.judge("Which port?", PASSAGES, "7420 [1]")
    assert first == second
    assert calls == [1]

    fresh, fresh_calls = make_judge([VALID], tmp_path)
    assert fresh.judge("Which port?", PASSAGES, "7420 [1]") == first
    assert fresh_calls == [0]

    fresh.judge("Which port?", PASSAGES, "a different answer")
    assert fresh_calls == [1]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_retries_once_on_invalid_reply(tmp_path: Path) -> None:
    judge, calls = make_judge(["not json", VALID], tmp_path)
    assert judge.judge("q", PASSAGES, "a").faithfulness == 5
    assert calls == [2]


def test_raises_after_exhausting_attempts_and_caches_nothing(tmp_path: Path) -> None:
    judge, calls = make_judge(["not json"], tmp_path)
    with pytest.raises(JudgeError, match="2 attempts"):
        judge.judge("q", PASSAGES, "a")
    assert calls == [2]
    assert list(tmp_path.iterdir()) == []


def test_works_without_cache() -> None:
    judge, calls = make_judge([VALID], None)
    judge.judge("q", PASSAGES, "a")
    judge.judge("q", PASSAGES, "a")
    assert calls == [2]
