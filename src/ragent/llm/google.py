from __future__ import annotations

import time
from copy import deepcopy
from typing import Any
from uuid import uuid4

import httpx

from ragent.errors import ProviderError

from .base import Completion, Message, ToolCall, Usage


def _google_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _google_schema(item)
            for key, item in value.items()
            if key != "additionalProperties"
        }
    if isinstance(value, list):
        return [_google_schema(item) for item in value]
    return value


class GoogleProvider:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.Client(timeout=httpx.Timeout(120.0))

    def _post(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/models/{model}:generateContent"
        last_error = "request failed"
        for attempt, delay in enumerate((0, 1, 4, 9)):
            if delay:
                time.sleep(delay)
            try:
                response = self.client.post(url, params={"key": self.api_key}, json=payload)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                if attempt < 3:
                    continue
                break
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if attempt < 3:
                    continue
                break
            if response.is_error:
                raise ProviderError(
                    f"Google provider request failed: HTTP {response.status_code}: {response.text[:500]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise ProviderError(f"Google returned invalid JSON: {exc}") from exc
        raise ProviderError(f"Google provider request failed after retries: {last_error}")

    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        system_parts = [message.content for message in messages if message.role == "system"]
        contents = [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in messages
            if message.role != "system"
        ]
        generation: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_schema is not None:
            generation.update(
                {
                    "responseMimeType": "application/json",
                    "responseSchema": _google_schema(deepcopy(json_schema)),
                }
            )
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if tools:
            declarations = []
            for tool in tools:
                function = tool.get("function", tool)
                declarations.append(
                    {
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "parameters": _google_schema(function.get("parameters", {})),
                    }
                )
            payload["tools"] = [{"functionDeclarations": declarations}]
        data = self._post(model, payload)
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Google returned an unexpected response: {data!r}") from exc
        text: list[str] = []
        calls: list[ToolCall] = []
        for part in parts:
            if "text" in part:
                text.append(str(part["text"]))
            if function := part.get("functionCall"):
                calls.append(
                    ToolCall(
                        id=f"call_{uuid4().hex[:12]}",
                        name=str(function.get("name", "")),
                        arguments=dict(function.get("args") or {}),
                    )
                )
        raw_usage = data.get("usageMetadata") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("promptTokenCount") or 0),
            completion_tokens=int(raw_usage.get("candidatesTokenCount") or 0),
            total_tokens=int(raw_usage.get("totalTokenCount") or 0),
            cost_usd=None,
        )
        return Completion(text="".join(text), tool_calls=calls, usage=usage)
