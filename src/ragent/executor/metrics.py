from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

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


def normalize_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    host = (parsed.hostname or "").lower()
    if parsed.port is not None:
        default = (parsed.scheme.lower() == "http" and parsed.port == 80) or (
            parsed.scheme.lower() == "https" and parsed.port == 443
        )
        if not default:
            host += f":{parsed.port}"
    if parsed.username or parsed.password:
        return ""
    return urlunsplit(
        (parsed.scheme.lower(), host, parsed.path or "/", parsed.query, "")
    )


def _artifact_file_exists(metric: Metric, artifact: str, ctx: RunContext) -> bool:
    if not metric.key.endswith("_path"):
        return bool(artifact.strip())
    if not artifact.strip():
        return False
    try:
        path = Path(artifact).expanduser().resolve(strict=True)
        run_root = ctx.run_dir.resolve(strict=True)
        roots = [run_root]
        if ctx.cfg.obsidian.vault_path is not None:
            roots.append(ctx.cfg.obsidian.vault_path.expanduser().resolve(strict=True))
        return (
            path.is_file()
            and path.stat().st_size > 0
            and any(path.is_relative_to(root) for root in roots)
        )
    except OSError:
        return False


def cited_source_urls(artifact: str, ctx: RunContext) -> set[str]:
    successful = {
        normalize_url(item.get("url", "")): normalize_url(item.get("url", ""))
        for item in ctx.citations
        if normalize_url(item.get("url", ""))
    }
    aliases = {
        normalize_url(requested): normalize_url(canonical)
        for requested, canonical in ctx.source_aliases.items()
        if normalize_url(requested) and normalize_url(canonical)
    }
    matched: set[str] = set()
    for raw in _URL.findall(artifact):
        normalized = normalize_url(raw.rstrip(".,;:"))
        canonical = aliases.get(normalized, normalized)
        if canonical in successful:
            matched.add(canonical)
    return matched


def check(metric: Metric, ctx: RunContext, cfg: Config | None = None) -> MetricResult:
    artifact = ctx.artifacts.get(metric.key, "")
    if metric.kind == "artifact_exists":
        passed = _artifact_file_exists(metric, artifact, ctx)
        return MetricResult(
            passed,
            f"artifact {metric.key} {'exists' if passed else 'is empty or unverified'}",
        )
    if metric.kind == "min_words":
        count = len(artifact.split())
        threshold = metric.n or 0
        return MetricResult(count >= threshold, f"{count} words; requires {threshold}")
    if metric.kind == "min_items":
        count = len(_ITEM.findall(artifact))
        threshold = metric.n or 0
        return MetricResult(
            count >= threshold, f"{count} list items; requires {threshold}"
        )
    if metric.kind == "has_citations":
        count = len(cited_source_urls(artifact, ctx))
        threshold = metric.n or 0
        return MetricResult(
            count >= threshold,
            f"{count} distinct fetched-and-cited sources; requires {threshold}",
        )
    if metric.kind == "regex":
        try:
            passed = bool(re.search(metric.pattern or "", artifact, re.MULTILINE))
        except re.error as exc:
            return MetricResult(False, f"invalid regex metric: {exc}")
        return MetricResult(
            passed, f"pattern {'matched' if passed else 'did not match'}"
        )
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
