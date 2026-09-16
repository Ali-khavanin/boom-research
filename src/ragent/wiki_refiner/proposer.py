from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ragent.config import Config
from ragent.graph_builder.schema import Edge, Metric, Node
from ragent.llm import get_llm
from ragent.llm.base import Message
from ragent.trajectories.store import stats


class ProposalOp(BaseModel):
    id: str
    op: Literal["add_edge", "update_edge", "update_metric", "add_node"]
    payload: dict[str, Any]
    rationale: str

    @model_validator(mode="after")
    def validate_payload(self) -> "ProposalOp":
        if self.op == "add_edge":
            Edge.model_validate(self.payload)
        elif self.op == "add_node":
            Node.model_validate(self.payload)
        elif self.op == "update_metric":
            if "edge_id" not in self.payload or "metric" not in self.payload:
                raise ValueError("update_metric requires edge_id and metric")
            Metric.model_validate(self.payload["metric"])
        elif self.op == "update_edge":
            if not self.payload.get("id") or not isinstance(self.payload.get("changes"), dict):
                raise ValueError("update_edge requires id and changes")
        return self


class Proposal(BaseModel):
    ops: list[ProposalOp] = Field(min_length=1)
    written_path: Path | None = Field(default=None, exclude=True)


def propose(cfg: Config) -> Proposal:
    evidence = stats(cfg.workspace)
    llm = get_llm("refiner", cfg=cfg)
    schema = Proposal.model_json_schema()
    raw = llm.json(
        [
            Message(
                "system",
                "Propose minimal graph maintenance from process-mining evidence. Never delete stages. "
                "Use unique snake_case operation ids. Prefer fixing a repeatedly failing edge or metric over adding nodes. "
                "Payloads must be complete for add operations and targeted for updates.",
            ),
            Message(
                "user",
                "Trajectory statistics and up to twelve concrete failure details follow. "
                "Return one or more evidence-backed operations.\n\n"
                + json.dumps(evidence, ensure_ascii=False, indent=2),
            ),
        ],
        schema,
    )
    proposal = Proposal.model_validate(raw)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = cfg.workspace / "refine" / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "proposal.json"
    path.write_text(proposal.model_dump_json(indent=2), encoding="utf-8")
    proposal.written_path = path
    return proposal
