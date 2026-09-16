from __future__ import annotations

from pathlib import Path

from .schema import Graph


def _label(node_id: str, title: str) -> str:
    text = f"{node_id} — {title}".replace('"', "#quot;")
    return " ".join(text.split("\n"))


def to_mermaid(graph: Graph) -> str:
    lines = ["flowchart TD", f"%% entry: {graph.entry}"]
    node_ids = {node.id for node in graph.nodes}
    for node in graph.nodes:
        label = _label(node.id, node.title)
        if node.terminal:
            lines.append(f'  {node.id}(["{label}"])')
        else:
            lines.append(f'  {node.id}["{label}"]')
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        lines.append(f"  {edge.source} -->|{edge.id}| {edge.target}")
    return "\n".join(lines) + "\n"


def write_mermaid(graph: Graph, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_mermaid(graph), encoding="utf-8")
    return path
