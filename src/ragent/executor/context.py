from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ragent.config import Config

if TYPE_CHECKING:
    from ragent.trajectories.log import TraceLog


@dataclass(slots=True)
class RunContext:
    run_id: str
    query: str
    workspace: Path
    artifacts: dict[str, str]
    citations: list[dict[str, str]]
    cfg: Config
    trace: TraceLog
    skill_text: str = ""
    source_aliases: dict[str, str] = field(default_factory=dict)
    attempts: Counter[str] = field(default_factory=Counter)
    last_failure: str = ""

    @property
    def run_dir(self) -> Path:
        return self.workspace / "runs" / self.run_id
