from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

CSV_FIELDS = ("case_id", "activity", "timestamp", "step", "edge_id", "metric_passed")


def load_runs(workspace: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    run_root = workspace / "runs"
    if not run_root.exists():
        return runs
    for trace_path in sorted(run_root.glob("*/trace.jsonl")):
        events: list[dict[str, Any]] = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        runs[trace_path.parent.name] = events
    return runs


def to_csv(workspace: Path, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for events in load_runs(workspace).values():
            for event in events:
                if "edge_id" not in event:
                    continue
                writer.writerow(
                    {
                        "case_id": event.get("case_id", ""),
                        "activity": event.get("activity", ""),
                        "timestamp": event.get("timestamp", ""),
                        "step": event.get("step", ""),
                        "edge_id": event.get("edge_id", ""),
                        "metric_passed": event.get("metric", {}).get("passed", False),
                    }
                )
    return path


def stats(workspace: Path) -> dict[str, Any]:
    edge_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incomplete: list[str] = []
    failures: list[dict[str, str]] = []
    for run_id, events in load_runs(workspace).items():
        metadata_path = workspace / "runs" / run_id / "run.json"
        explicit_status: str | None = None
        if metadata_path.is_file():
            try:
                value = json.loads(metadata_path.read_text(encoding="utf-8"))
                explicit_status = str(value.get("status", ""))
            except (OSError, json.JSONDecodeError, AttributeError):
                explicit_status = "failed"
        reached_done = explicit_status == "completed"
        for event in events:
            if edge_id := event.get("edge_id"):
                edge_events[edge_id].append(event)
                metric = event.get("metric", {})
                if not metric.get("passed", False):
                    failures.append(
                        {
                            "run_id": run_id,
                            "edge_id": edge_id,
                            "detail": str(metric.get("detail", "")),
                        }
                    )
                if (
                    explicit_status is None
                    and event.get("target") == "done"
                    and metric.get("passed", False)
                ):
                    reached_done = True
        if not reached_done:
            incomplete.append(run_id)
    per_edge = {}
    for edge_id, events in sorted(edge_events.items()):
        passed = sum(bool(event.get("metric", {}).get("passed")) for event in events)
        per_edge[edge_id] = {
            "attempts": len(events),
            "pass_rate": passed / len(events),
            "mean_artifact_size": mean(
                event.get("artifact_chars", 0) for event in events
            ),
        }
    return {"edges": per_edge, "incomplete_runs": incomplete, "failures": failures[:12]}
