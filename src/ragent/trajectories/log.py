from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class TraceLog:
    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / "trace.jsonl"
        self.on_event = on_event
        self.step = 0
        run_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        event = {
            "case_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": self.step,
            **event,
        }
        self.step += 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(event)
        return event

    def transition(
        self,
        *,
        node_id: str,
        edge_id: str,
        target: str,
        metric: dict[str, Any],
        artifact_key: str,
        artifact_chars: int,
        tool_calls: list[str],
        model: str,
        attempt: int,
        tokens: int = 0,
        cost_usd: float | None = None,
    ) -> dict[str, Any]:
        return self.append(
            {
                "activity": node_id,
                "edge_id": edge_id,
                "target": target,
                "metric": metric,
                "artifact_key": artifact_key,
                "artifact_chars": artifact_chars,
                "tool_calls": tool_calls,
                "model": model,
                "attempt": attempt,
                "tokens": tokens,
                "cost_usd": cost_usd,
            }
        )

    def error(self, *, node_id: str, detail: str) -> dict[str, Any]:
        return self.append({"activity": node_id, "event": "error", "detail": detail})
