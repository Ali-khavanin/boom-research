from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import BaseModel, Field

from .schema import Graph


class Finding(BaseModel):
    level: Literal["error", "info"]
    code: str
    detail: str


class Audit(BaseModel):
    entry: str
    node_count: int
    done_reachable_count: int
    dead_ends: list[str] = Field(default_factory=list)
    unreachable: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(finding.level == "error" for finding in self.findings)


def audit(graph: Graph) -> Audit:
    ids = {node.id for node in graph.nodes}
    findings: list[Finding] = []
    if len(ids) != len(graph.nodes):
        findings.append(Finding(level="error", code="duplicate_node", detail="node ids are not unique"))
    edge_ids = {edge.id for edge in graph.edges}
    if len(edge_ids) != len(graph.edges):
        findings.append(Finding(level="error", code="duplicate_edge", detail="edge ids are not unique"))
    if graph.entry not in ids:
        findings.append(Finding(level="error", code="missing_entry", detail=f"entry id is unresolved: {graph.entry}"))
    terminals = [node.id for node in graph.nodes if node.terminal]
    if len(terminals) != 1:
        findings.append(Finding(level="error", code="terminal_count", detail=f"expected exactly one terminal node, found {len(terminals)}"))
    for edge in graph.edges:
        if edge.source not in ids:
            findings.append(Finding(level="error", code="bad_source", detail=f"edge {edge.id} source is unresolved: {edge.source}"))
        if edge.target not in ids:
            findings.append(Finding(level="error", code="bad_target", detail=f"edge {edge.id} target is unresolved: {edge.target}"))
        if edge.on_fail and edge.on_fail not in ids:
            findings.append(Finding(level="error", code="bad_on_fail", detail=f"edge {edge.id} on_fail is unresolved: {edge.on_fail}"))
    dead_ends = sorted(
        node.id for node in graph.nodes if not node.terminal and not graph.out_edges(node.id)
    )
    for node_id in dead_ends:
        findings.append(Finding(level="error", code="dead_end", detail=f"non-terminal node has no outward edge: {node_id}"))
    reachable: set[str] = set()
    if graph.entry in ids:
        queue = deque([graph.entry])
        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            queue.extend(edge.target for edge in graph.out_edges(node_id) if edge.target in ids)
    unreachable = sorted(ids - reachable)
    for node_id in unreachable:
        findings.append(Finding(level="error", code="unreachable", detail=f"node is unreachable from entry: {node_id}"))
    reverse: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for edge in graph.edges:
        if edge.source in ids and edge.target in ids:
            reverse[edge.target].append(edge.source)
    reaches_done: set[str] = set()
    if "done" in ids:
        queue = deque(["done"])
        while queue:
            node_id = queue.popleft()
            if node_id in reaches_done:
                continue
            reaches_done.add(node_id)
            queue.extend(reverse[node_id])
    for node_id in sorted(ids - reaches_done):
        findings.append(Finding(level="error", code="cannot_reach_done", detail=f"done is not reachable from node: {node_id}"))
    if terminals and (not reachable or terminals[0] not in reachable):
        findings.append(Finding(level="error", code="terminal_unreachable", detail=f"terminal node is unreachable: {terminals[0]}"))
    if not findings:
        findings.append(Finding(level="info", code="sound", detail="graph reachability and dead-end checks passed"))
    return Audit(
        entry=graph.entry,
        node_count=len(graph.nodes),
        done_reachable_count=len(reaches_done),
        dead_ends=dead_ends,
        unreachable=unreachable,
        findings=findings,
    )
