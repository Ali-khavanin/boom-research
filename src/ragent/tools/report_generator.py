from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ragent.errors import RagentError
from ragent.executor.metrics import cited_source_urls, normalize_url
from ragent.llm import get_llm
from ragent.llm.base import Message

if TYPE_CHECKING:
    from ragent.executor.context import RunContext

_SECTIONS = (
    ("Goal", "goal"),
    ("Prior Work", "prior_work"),
    ("Limitations", "limitations"),
    ("Gaps", "gaps"),
    ("Feasibility", "feasibility"),
    ("Quick Test", "quick_test"),
)
_URL = re.compile(r"https?://[^\s)\]>\"']+")


def _canonical(ctx: RunContext, value: str) -> str:
    normalized = normalize_url(value.rstrip(".,;:"))
    return ctx.source_aliases.get(normalized, normalized)


def _successful(ctx: RunContext) -> set[str]:
    return {
        normalize_url(item.get("url", ""))
        for item in ctx.citations
        if normalize_url(item.get("url", ""))
    }


def _validate_report(ctx: RunContext, report: str) -> None:
    required_headings = [f"## {heading}" for heading, _ in _SECTIONS] + [
        "## References"
    ]
    missing = [heading for heading in required_headings if heading not in report]
    if missing:
        raise RagentError("report is missing required headings: " + ", ".join(missing))
    successful = _successful(ctx)
    unknown = sorted(
        {raw for raw in _URL.findall(report) if _canonical(ctx, raw) not in successful}
    )
    if unknown:
        raise RagentError(
            "report contains unfetched source URLs: " + ", ".join(unknown)
        )
    cited = cited_source_urls(report, ctx)
    if len(cited) < ctx.cfg.research.min_sources:
        raise RagentError(
            f"report cites {len(cited)} fetched sources; requires {ctx.cfg.research.min_sources}"
        )


def generate(ctx: RunContext) -> str:
    cached = ctx.artifacts.get("report_path", "")
    if cached:
        path = Path(cached)
        if path.is_file() and path.stat().st_size:
            report = path.read_text(encoding="utf-8")
            _validate_report(ctx, report)
            return str(path)
    missing = [key for _, key in _SECTIONS if not ctx.artifacts.get(key, "").strip()]
    if missing:
        raise RagentError(
            "cannot generate report; missing stage artifacts: " + ", ".join(missing)
        )
    stage_text = "\n\n".join(ctx.artifacts[key] for _, key in _SECTIONS)
    cited = cited_source_urls(stage_text, ctx)
    if len(cited) < ctx.cfg.research.min_sources:
        raise RagentError(
            f"stage artifacts cite {len(cited)} fetched sources; requires {ctx.cfg.research.min_sources}"
        )
    sections = [
        f"# Research Report\n\n**Query:** {ctx.query}",
        "## Review Method\n\n"
        + (
            ctx.cfg.research.instructions.strip()
            or "Narrative review using fetched source evidence."
        ),
    ]
    for heading, key in _SECTIONS:
        sections.append(f"## {heading}\n\n{ctx.artifacts[key]}")
    references = []
    seen: set[str] = set()
    for citation in ctx.citations:
        url = normalize_url(citation.get("url", ""))
        if url and url not in seen and url in cited:
            seen.add(url)
            provenance = citation.get("fetched_at", "")
            references.append(
                f"- [{citation.get('title') or url}]({url}) — retrieved {provenance}"
            )
    sections.append("## References\n\n" + "\n".join(references))
    draft = "\n\n".join(sections).strip() + "\n"
    llm = get_llm("report", cfg=ctx.cfg)
    polished = llm.text(
        [
            Message(
                "system",
                "Edit the supplied report for clarity and cohesion. Preserve every heading, fact, "
                "URL, qualification, review instruction, and omission. Never introduce a new fact "
                "or citation. Return only Markdown.",
            ),
            Message("user", draft),
        ]
    ).strip()
    use_polished = bool(polished)
    if use_polished:
        try:
            _validate_report(ctx, polished)
            draft_urls = {_canonical(ctx, value) for value in _URL.findall(draft)}
            polished_urls = {_canonical(ctx, value) for value in _URL.findall(polished)}
            use_polished = polished_urls == draft_urls
        except RagentError:
            use_polished = False
    report = (polished if use_polished else draft).strip() + "\n"
    _validate_report(ctx, report)
    path = ctx.run_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    ctx.artifacts["report_path"] = str(path)
    return str(path)
