from __future__ import annotations

import re
from dataclasses import dataclass

from ragent.config import Config
from ragent.graph_builder.schema import Metric
from ragent.llm import get_llm
from ragent.llm.base import Message

from .context import RunContext

_ITEM = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_URL = re.compile(r"https?://[^\s)\]>\"']+")


@dataclass(slots=True)
class MetricResult:
    passed: bool
    detail: str


def check(metric: Metric, ctx: RunContext, cfg: Config | None = None) -> MetricResult:
    artifact = ctx.artifacts.get(metric.key, "")
    if metric.kind == "artifact_exists":
        passed = bool(artifact.strip())
        return MetricResult(passed, f"artifact {metric.key} {'exists' if passed else 'is empty'}")
    if metric.kind == "min_words":
        count = len(artifact.split())
        threshold = metric.n or 0
        return MetricResult(count >= threshold, f"{count} words; requires {threshold}")
    if metric.kind == "min_items":
        count = len(_ITEM.findall(artifact))
        threshold = metric.n or 0
        return MetricResult(count >= threshold, f"{count} list items; requires {threshold}")
    if metric.kind == "has_citations":
        count = len(set(_URL.findall(artifact)))
        threshold = metric.n or 0
        return MetricResult(count >= threshold, f"{count} distinct cited URLs; requires {threshold}")
    if metric.kind == "regex":
        try:
            passed = bool(re.search(metric.pattern or "", artifact, re.MULTILINE))
        except re.error as exc:
            return MetricResult(False, f"invalid regex metric: {exc}")
        return MetricResult(passed, f"pattern {'matched' if passed else 'did not match'}")
    llm = get_llm("judge", cfg=cfg or ctx.cfg)
    schema = {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["passed", "reason"],
        "additionalProperties": False,
    }
    result = llm.json(
        [
            Message(
                "system",
                "Judge only whether the artifact satisfies the rubric. Do not infer from history or claims of progress.",
            ),
            Message("user", f"Rubric:\n{metric.rubric}\n\nArtifact:\n{artifact}"),
        ],
        schema,
    )
    return MetricResult(bool(result["passed"]), str(result["reason"]))
