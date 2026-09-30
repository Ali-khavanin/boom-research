from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ragent.book_to_skill import load_skill
from ragent.errors import GraphError
from ragent.llm.base import LLM, Message

from .schema import Edge, Graph, Node, graph_schema, slug_id
from .seed import load_seed
from .validate import audit


_GRAPH_KEYS = {"version", "entry", "nodes", "edges"}
_NODE_KEYS = {"id", "title", "description", "terminal", "model", "provenance"}
_ROLE_OVERRIDE_KEYS = {"provider", "model", "temperature", "max_tokens"}
_EDGE_KEYS = {
    "id", "source", "target", "prompt_template", "tool_set", "precondition",
    "termination_metric", "produces", "on_fail", "max_attempts", "provenance",
}
_METRIC_KEYS = {"kind", "key", "n", "pattern", "rubric"}


def _keep(value: dict, keys: set[str]) -> dict:
    return {k: v for k, v in value.items() if k in keys}


def _sanitize_graph_proposal(value: dict[str, Any]) -> dict[str, Any]:
    """Strip properties the model hallucinated beyond the Graph schema (some models
    ignore `additionalProperties: false` and add stray fields like a `kind` discriminator
    on redeclared seed nodes)."""
    out = _keep(value, _GRAPH_KEYS)
    nodes = []
    for node in out.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        clean = _keep(node, _NODE_KEYS)
        if isinstance(clean.get("model"), dict):
            clean["model"] = _keep(clean["model"], _ROLE_OVERRIDE_KEYS)
        nodes.append(clean)
    out["nodes"] = nodes
    edges = []
    for edge in out.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        clean = _keep(edge, _EDGE_KEYS)
        for metric_key in ("precondition", "termination_metric"):
            if isinstance(clean.get(metric_key), dict):
                clean[metric_key] = _keep(clean[metric_key], _METRIC_KEYS)
        edges.append(clean)
    out["edges"] = edges
    return out


def _normalize_node(node: Node, chapter: str) -> Node:
    original = node.id
    node.id = slug_id(node.id)
    node.provenance = {"chapter": chapter, "cue": node.provenance.get("cue", "") if node.provenance else ""}
    if not node.title:
        node.title = original
    return node


def _normalize_edge(edge: Edge, chapter: str, id_map: dict[str, str]) -> Edge:
    edge.id = slug_id(edge.id)
    edge.source = id_map.get(edge.source, slug_id(edge.source))
    edge.target = id_map.get(edge.target, slug_id(edge.target))
    if edge.on_fail:
        edge.on_fail = id_map.get(edge.on_fail, slug_id(edge.on_fail))
    edge.provenance = {"chapter": chapter, "cue": edge.provenance.get("cue", "") if edge.provenance else ""}
    return edge


def build_graph(
    skill_dir: Path,
    llm: LLM,
    seed: Graph | None = None,
    *,
    extend: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Graph:
    def _emit(payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(payload)
        except Exception:
            pass

    base = (seed or load_seed()).model_copy(deep=True)
    if not extend:
        return base
    node_ids = {node.id for node in base.nodes}
    edge_ids = {edge.id for edge in base.edges}
    pairs = {(edge.source, edge.target) for edge in base.edges}
    files = load_skill(skill_dir).chapter_files
    _emit({"event": "graph_start", "chapters": len(files)})
    for position, path in enumerate(files):
        chapter = path.name
        source = path.read_text(encoding="utf-8")
        _emit(
            {
                "event": "graph_chapter_start",
                "chapter": chapter,
                "index": position + 1,
                "total": len(files),
            }
        )
        prompt = [
            Message(
                "system",
                "Compile one chapter file of a book-to-skill generated skill into a sound research-stage graph. "
                "The chapter follows the book-to-skill template: Core Idea, Frameworks Introduced (each with "
                "When to use and How), Key Concepts, Mental Models, Anti-patterns, optional Worked Example, "
                "Key Takeaways, and Connects To. Treat each framework's How steps as ordered stages, its "
                "When to use as the edge precondition or prompt intent, Anti-patterns as failure conditions "
                "that justify on_fail loop-backs, and Key Takeaways as termination criteria. "
                "Reuse the seed ids start, goal, what_has_been_done, limitations, gaps, feasibility, quick_test, "
                "done whenever applicable. Add a node only for a genuinely new stage. Every non-seed node "
                "and edge needs provenance with chapter and cue, where cue is an exact phrase copied from the chapter text. "
                "Metric field rules: kinds min_words, min_items, and has_citations each require an integer n "
                "(e.g. has_citations needs n = minimum citation count); kind regex requires pattern; "
                "kind llm_rubric requires rubric. "
                "Connectivity is mandatory: your JSON is merged as-is, with no cross-chapter wiring added "
                "afterward. Every node you include (seed or new) must sit on at least one edge path that "
                "starts at a seed node reachable from 'start' and ends at 'done'. Never add a node with no "
                "outgoing edge unless it is 'done' itself; never add a node with no incoming edge from "
                "'start' or another node already on such a path. If a new node does not chain forward to "
                "'done', omit it rather than leave it disconnected.",
            ),
            Message(
                "user",
                f"Existing graph node ids already merged from other chapters (for id reuse only, do not "
                f"redefine): {sorted(node_ids)}\n\n"
                f"Chapter: {chapter}\nReturn a complete Graph-shaped JSON proposal. It may repeat seed elements; "
                f"deterministic merging will keep seed definitions.\n\n{source}",
            ),
        ]
        proposal = Graph.model_validate(
            llm.json(
                prompt,
                graph_schema(),
                validate_fn=lambda value: Graph.model_validate(value),
                sanitize_fn=_sanitize_graph_proposal,
            )
        )
        id_map = {node.id: slug_id(node.id) for node in proposal.nodes}
        added_nodes: set[str] = set()
        added_edges: set[str] = set()
        new_nodes: list[dict[str, str]]
        new_edges: list[dict[str, str]]
        for proposed in proposal.nodes:
            node = _normalize_node(proposed, chapter)
            if node.id not in node_ids:
                base.nodes.append(node)
                node_ids.add(node.id)
                added_nodes.add(node.id)
        for proposed in proposal.edges:
            edge = _normalize_edge(proposed, chapter, id_map)
            pair = (edge.source, edge.target)
            if edge.source not in node_ids or edge.target not in node_ids:
                continue
            if edge.id in edge_ids or pair in pairs:
                continue
            base.edges.append(edge)
            edge_ids.add(edge.id)
            pairs.add(pair)
            added_edges.add(edge.id)
        # Deterministically prune this chapter's additions until the graph is sound:
        # base was sound before this chapter (seed and every prior chapter already passed
        # this same check), so any error finding after merging must implicate a node or
        # edge added in THIS chapter. Iterate to a fixed point rather than trusting the
        # model's own connectivity claims.
        for _ in range(len(added_nodes) + len(added_edges) + 1):
            result = audit(base)
            if result.ok:
                break
            prunable_nodes = {
                finding.detail.rsplit(": ", 1)[-1]
                for finding in result.findings
                if finding.code in {"dead_end", "unreachable", "cannot_reach_done"}
            } & added_nodes
            if not prunable_nodes:
                break
            base.nodes = [node for node in base.nodes if node.id not in prunable_nodes]
            base.edges = [
                edge
                for edge in base.edges
                if edge.source not in prunable_nodes and edge.target not in prunable_nodes
            ]
            pruned_edges = added_edges - {edge.id for edge in base.edges}
            node_ids -= prunable_nodes
            edge_ids -= pruned_edges
            pairs = {(edge.source, edge.target) for edge in base.edges}
            added_nodes -= prunable_nodes
            added_edges -= pruned_edges
        new_nodes = [{"id": node.id, "title": node.title} for node in base.nodes if node.id in added_nodes]
        new_edges = [
            {"id": edge.id, "source": edge.source, "target": edge.target}
            for edge in base.edges
            if edge.id in added_edges
        ]
        _emit(
            {
                "event": "graph_delta",
                "chapter": chapter,
                "index": position + 1,
                "total": len(files),
                "new_nodes": new_nodes,
                "new_edges": new_edges,
                "node_count": len(base.nodes),
                "edge_count": len(base.edges),
            }
        )
    result = audit(base)
    detail = "; ".join(finding.detail for finding in result.findings if finding.level == "error")
    _emit(
        {
            "event": "graph_complete",
            "audit_ok": result.ok,
            "detail": detail,
            "graph": base.model_dump(mode="json"),
        }
    )
    if not result.ok:
        raise GraphError(f"compiled graph failed audit: {detail}")
    return base
