from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragent.config import Config
from ragent.errors import GraphError
from ragent.executor.runner import run
from ragent.graph_builder.schema import Graph
from ragent.graph_builder.validate import audit

from .evalset import load_eval_set


@dataclass(slots=True)
class Scorecard:
    success: float
    quality: float
    queries: int


def _score(graph: Graph, cfg: Config) -> Scorecard:
    successes = 0
    first_passes = 0
    metric_events = 0
    for query in load_eval_set():
        events: list[dict[str, Any]] = []
        try:
            result = run(
                graph, query, cfg, report=False, obsidian=False, on_event=events.append
            )
            successes += int(result.reached_done)
        except Exception:
            pass
        for event in events:
            metric = event.get("metric")
            if not isinstance(metric, dict):
                continue
            metric_events += 1
            if event.get("attempt") == 1 and metric.get("passed", False):
                first_passes += 1
    count = len(load_eval_set())
    return Scorecard(
        success=successes / count if count else 0.0,
        quality=first_passes / metric_events if metric_events else 0.0,
        queries=count,
    )


def _latest_snapshot(workspace: Path) -> Path | None:
    versions = sorted((workspace / "graph_versions").glob("*/graph.json"))
    return versions[-1] if versions else None


def gated(cfg: Config, candidate_graph: Path | Graph) -> dict[str, Any]:
    current_path = cfg.workspace / "graph.json"
    if not current_path.exists():
        raise GraphError(f"current graph not found: {current_path}")
    current = Graph.model_validate_json(current_path.read_text(encoding="utf-8"))
    candidate = (
        candidate_graph
        if isinstance(candidate_graph, Graph)
        else Graph.model_validate_json(candidate_graph.read_text(encoding="utf-8"))
    )
    candidate_audit = audit(candidate)
    if not candidate_audit.ok:
        details = "; ".join(
            item.detail for item in candidate_audit.findings if item.level == "error"
        )
        raise GraphError(f"candidate graph failed audit: {details}")
    baseline_score = _score(current, cfg)
    candidate_score = _score(candidate, cfg)
    kept = (
        candidate_score.success >= baseline_score.success
        and candidate_score.quality >= baseline_score.quality - 0.02
    )
    if kept:
        current_path.write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
    else:
        snapshot = _latest_snapshot(cfg.workspace)
        if snapshot is not None:
            shutil.copy2(snapshot, current_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = cfg.workspace / "refine" / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "kept": kept,
        "baseline": asdict(baseline_score),
        "candidate": asdict(candidate_score),
        "threshold": {"success_drop": 0.0, "quality_drop": 0.02},
    }
    (directory / "rollback.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record
