from __future__ import annotations

import json
from typing import Any, Callable

from jsonschema import ValidationError, validate

from ragent.errors import ProviderError

from .base import LLM, Message


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def call_json(
    llm: LLM,
    messages: list[Message],
    schema: dict[str, Any],
    retries: int = 2,
    validate_fn: Callable[[dict[str, Any]], None] | None = None,
    sanitize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    current = list(messages)
    last_error = "invalid JSON"
    for attempt in range(retries + 1):
        completion = llm.complete(current, json_schema=schema)
        try:
            value = json.loads(_strip_fences(completion.text))
            if not isinstance(value, dict):
                raise ValueError("top-level value must be an object")
            if sanitize_fn is not None:
                value = sanitize_fn(value)
            validate(instance=value, schema=schema)
            if validate_fn is not None:
                validate_fn(value)
            return value
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = str(exc)
            if attempt < retries:
                current = [
                    *messages,
                    Message(
                        "user",
                        "Return only corrected JSON matching the supplied schema. "
                        f"Previous validation error: {last_error}",
                    ),
                ]
    raise ProviderError(
        f"{llm.role} provider returned invalid structured output after {retries + 1} attempts: {last_error}"
    )
