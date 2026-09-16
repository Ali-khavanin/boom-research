from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

if TYPE_CHECKING:
    from .usage import UsageLedger


@dataclass(slots=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass(slots=True)
class Completion:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage = field(default_factory=Usage)


class Provider(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion: ...


@dataclass(slots=True)
class LLM:
    provider: Provider
    model: str
    temperature: float
    max_tokens: int
    role: str
    provider_name: str = ""
    ledger: "UsageLedger | None" = None

    @property
    def label(self) -> str:
        return f"{self.provider_name}/{self.model}" if self.provider_name else self.model

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        if self.ledger is not None:
            self.ledger.precheck(role=self.role, label=self.label)
        completion = self.provider.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_schema=json_schema,
            tools=tools,
        )
        if self.ledger is not None:
            self.ledger.record(role=self.role, label=self.label, usage=completion.usage)
        return completion

    def text(self, messages: list[Message]) -> str:
        return self.complete(messages).text

    def json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        validate_fn: Callable[[dict[str, Any]], None] | None = None,
        sanitize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from .structured import call_json

        return call_json(self, messages, schema, validate_fn=validate_fn, sanitize_fn=sanitize_fn)
