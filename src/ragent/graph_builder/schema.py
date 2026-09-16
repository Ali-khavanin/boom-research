from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragent.errors import GraphError


class RoleOverride(BaseModel):
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class Metric(BaseModel):
    kind: Literal[
        "artifact_exists",
        "min_words",
        "min_items",
        "has_citations",
        "regex",
        "llm_rubric",
    ]
    key: str = "output"
    n: int | None = None
    pattern: str | None = None
    rubric: str | None = None

    @model_validator(mode="after")
    def validate_parameters(self) -> "Metric":
        if self.kind in {"min_words", "min_items", "has_citations"} and self.n is None:
            raise ValueError(f"metric {self.kind} requires n")
        if self.kind == "regex" and not self.pattern:
            raise ValueError("regex metric requires pattern")
        if self.kind == "llm_rubric" and not self.rubric:
            raise ValueError("llm_rubric metric requires rubric")
        return self


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    target: str
    prompt_template: str
    tool_set: list[str] = Field(default_factory=list)
    precondition: Metric | None = None
    termination_metric: Metric
    produces: str = "output"
    on_fail: str | None = None
    max_attempts: int = Field(default=2, ge=1)
    provenance: dict[str, Any] | None = None


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str = ""
    terminal: bool = False
    model: RoleOverride | None = None
    provenance: dict[str, Any] | None = None


class Graph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    entry: str = "start"
    nodes: list[Node]
    edges: list[Edge]

    def out_edges(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise GraphError(f"graph node does not exist: {node_id}")


def graph_schema() -> dict[str, Any]:
    return Graph.model_json_schema()


def slug_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "stage"
