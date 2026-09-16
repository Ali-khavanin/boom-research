from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ragent.errors import GraphError

from . import browser, obsidian_mcp, report_generator


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[..., Any]

    @property
    def wire_name(self) -> str:
        return self.name.replace(".", "__")

    def wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.wire_name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


def _catalog(ctx: Any) -> dict[str, ToolSpec]:
    return {
        "browser.search": ToolSpec(
            "browser.search",
            "Search the web for relevant sources.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
            lambda **kwargs: browser.search(ctx, **kwargs),
        ),
        "browser.fetch": ToolSpec(
            "browser.fetch",
            "Fetch and extract readable text from a URL; fetched URLs become citations.",
            {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
            lambda **kwargs: browser.fetch(ctx, **kwargs),
        ),
        "report.generate": ToolSpec(
            "report.generate",
            "Generate the final report from all stage artifacts and citations.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda **_: {"report_path": report_generator.generate(ctx)},
        ),
        "obsidian.note": ToolSpec(
            "obsidian.note",
            "Write a research note to the configured Obsidian backend.",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
            lambda **kwargs: {"note_path": obsidian_mcp.note(ctx, **kwargs)},
        ),
    }


def bind(names: list[str], ctx: Any) -> list[ToolSpec]:
    catalog = _catalog(ctx)
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise GraphError(f"unknown graph tool(s): {', '.join(unknown)}")
    return [catalog[name] for name in names]
