from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ragent.errors import ProviderError

from .base import Completion, Message, ToolCall, Usage


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        extra_headers: dict[str, str] | None = None,
        report_cost: bool = False,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        headers.update(extra_headers or {})
        self.report_cost = report_cost
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(120.0),
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error = "request failed"
        for attempt, delay in enumerate((0, 1, 4, 9)):
            if delay:
                time.sleep(delay)
            try:
                response = self.client.post("/chat/completions", json=payload)
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
                    f"provider request failed: HTTP {response.status_code}: {response.text[:500]}"
                )
            try:
                return response.json()
            except ValueError as exc:
                raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
        raise ProviderError(f"provider request failed after retries: {last_error}")

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
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": json_schema},
            }
            if self.report_cost:
                # OpenRouter reasoning models can spend the entire completion budget on
                # chain-of-thought before ever emitting the JSON payload, returning empty
                # content. Structured output does not benefit from unbounded reasoning, so
                # cap it well below max_tokens to leave room for the actual payload.
                # Some models reject `enabled: false` ("mandatory reasoning"), so bound
                # the reasoning budget instead of disabling it outright.
                payload["reasoning"] = {"max_tokens": max(256, max_tokens // 4)}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.report_cost:
            payload["usage"] = {"include": True}
        try:
            data = self._post(payload)
        except ProviderError as exc:
            if json_schema is None or "400" not in str(exc):
                raise
            payload.pop("response_format", None)
            payload["messages"].append(
                {
                    "role": "user",
                    "content": "Return only JSON matching this schema: " + json.dumps(json_schema),
                }
            )
            data = self._post(payload)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"provider returned an unexpected response: {data!r}") from exc
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            calls.append(
                ToolCall(
                    id=str(raw.get("id", f"call_{len(calls)}")),
                    name=str(function.get("name", "")),
                    arguments=arguments,
                )
            )
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        raw_usage = data.get("usage") or {}
        cost = raw_usage.get("cost")
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
            completion_tokens=int(raw_usage.get("completion_tokens") or 0),
            total_tokens=int(raw_usage.get("total_tokens") or 0),
            cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        )
        return Completion(text=str(content), tool_calls=calls, usage=usage)
