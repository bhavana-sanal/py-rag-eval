import json

import httpx

from rag_eval.generation import (
    NO_ANSWER,
    ExtractiveGenerator,
    LLMGenerator,
    build_messages,
    parse_citations,
)
from rag_eval.llm import ChatClient
from rag_eval.schema import Chunk

PASSAGES = [
    Chunk("a#0", "a", "The server listens on port 7420."),
    Chunk("b#0", "b", "Retries back off.", context="Jobs fail. Retries back off exponentially."),
]


def test_parse_citations_handles_groups_and_duplicates() -> None:
    assert parse_citations("Yes [2]. Also [1, 3] and again [2].") == (2, 1, 3)
    assert parse_citations("no markers, [x] or [] here") == ()


def test_prompt_numbers_passages_and_uses_their_context_window() -> None:
    messages = build_messages("Which port?", PASSAGES)
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "[1] The server listens on port 7420." in user
    assert "[2] Jobs fail. Retries back off exponentially." in user
    assert user.endswith("Question: Which port?")


def test_extractive_generator_cites_the_best_matching_sentence() -> None:
    answer = ExtractiveGenerator().generate("How do retries back off?", PASSAGES)
    assert answer.text == "Retries back off exponentially. [2]"
    assert answer.citations == (2,)


def test_extractive_generator_abstains_without_overlap() -> None:
    answer = ExtractiveGenerator().generate("Who won the match?", PASSAGES)
    assert answer.text == NO_ANSWER
    assert answer.citations == ()


def test_llm_generator_calls_chat_completions() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = {"choices": [{"message": {"content": "Port 7420 [1]."}}]}
        return httpx.Response(200, json=reply)

    client = ChatClient(
        "llama3.3",
        base_url="http://llm.test/v1/",
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    answer = LLMGenerator(client).generate("Which port?", PASSAGES)

    assert answer.citations == (1,)
    (request,) = requests
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "llama3.3"
    assert body["temperature"] == 0.0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
