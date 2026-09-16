from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ragent.config import Config, load_config
from ragent.errors import GraphError
from ragent.graph_builder.schema import Edge, Graph, Metric, Node
from ragent.graph_builder.validate import audit

from .proposer import Proposal


def load_proposal(path: Path) -> Proposal:
    return Proposal.model_validate_json(path.read_text(encoding="utf-8"))


def merge(
    proposal: Proposal,
    accept: list[str],
    cfg: Config | None = None,
) -> Path:
    config = cfg or load_config()
    graph_path = config.workspace / "graph.json"
    if not graph_path.exists():
        raise GraphError(f"current graph not found: {graph_path}")
    graph = Graph.model_validate_json(graph_path.read_text(encoding="utf-8"))
    selected = proposal.ops if "all" in accept else [op for op in proposal.ops if op.id in set(accept)]
    if not selected:
        raise GraphError("no proposal operations were selected")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = config.workspace / "graph_versions" / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(graph_path, version_dir / "graph.json")
    for operation in selected:
        payload = operation.payload
        if operation.op == "add_node":
            node = Node.model_validate(payload)
            if any(existing.id == node.id for existing in graph.nodes):
                raise GraphError(f"add_node id already exists: {node.id}")
            graph.nodes.append(node)
        elif operation.op == "add_edge":
            edge = Edge.model_validate(payload)
            if any(existing.id == edge.id for existing in graph.edges):
                raise GraphError(f"add_edge id already exists: {edge.id}")
            graph.edges.append(edge)
        elif operation.op == "update_metric":
            edge_id = str(payload["edge_id"])
            edge = next((item for item in graph.edges if item.id == edge_id), None)
            if edge is None:
                raise GraphError(f"update_metric edge not found: {edge_id}")
            edge.termination_metric = Metric.model_validate(payload["metric"])
        else:
            edge_id = str(payload["id"])
            index = next((i for i, item in enumerate(graph.edges) if item.id == edge_id), None)
            if index is None:
                raise GraphError(f"update_edge edge not found: {edge_id}")
            data = graph.edges[index].model_dump()
            data.update(payload["changes"])
            graph.edges[index] = Edge.model_validate(data)
    result = audit(graph)
    if not result.ok:
        details = "; ".join(item.detail for item in result.findings if item.level == "error")
        raise GraphError(f"proposal regresses graph audit: {details}")
    graph_path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    return graph_path
