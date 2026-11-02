"""Minimal client for any OpenAI-compatible ``/chat/completions`` endpoint.

Works with hosted APIs and with local servers such as Ollama
(``base_url = "http://localhost:11434/v1"``), vLLM or llama.cpp.
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel

DEFAULT_BASE_URL = "https://api.openai.com/v1"

Message = dict[str, str]


class _ChatMessage(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _ChatMessage


class _ChatResponse(BaseModel):
    choices: list[_Choice]


class ChatClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Transport-level retries cover dropped connections, the common failure with
        # local model servers that are still loading weights.
        self._http = http_client or httpx.Client(
            timeout=timeout, transport=httpx.HTTPTransport(retries=2)
        )

    def complete(self, messages: list[Message], *, temperature: float = 0.0) -> str:
        response = self._http.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": temperature},
            headers=self._headers,
        )
        response.raise_for_status()
        parsed = _ChatResponse.model_validate_json(response.content)
        if not parsed.choices:
            raise ValueError("chat completion returned no choices")
        return parsed.choices[0].message.content
