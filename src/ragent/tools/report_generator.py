from __future__ import annotations

import re
from typing import TYPE_CHECKING

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


def generate(ctx: "RunContext") -> str:
    sections = [f"# Research Report\n\n**Query:** {ctx.query}"]
    for heading, key in _SECTIONS:
        sections.append(f"## {heading}\n\n{ctx.artifacts.get(key, '_Not produced._')}")
    references = []
    seen: set[str] = set()
    for citation in ctx.citations:
        url = citation.get("url", "")
        if url and url not in seen:
            seen.add(url)
            references.append(f"- [{citation.get('title') or url}]({url})")
    sections.append("## References\n\n" + ("\n".join(references) or "_No fetched sources._"))
    draft = "\n\n".join(sections).strip() + "\n"
    llm = get_llm("report", cfg=ctx.cfg)
    polished = llm.text(
        [
            Message(
                "system",
                "Edit the supplied report for clarity and cohesion. Preserve every heading, fact, "
                "URL, qualification, and omission. Never introduce a new fact or citation. Return only Markdown.",
            ),
            Message("user", draft),
        ]
    ).strip()
    required_headings = [f"## {heading}" for heading, _ in _SECTIONS] + ["## References"]
    polished_urls = set(re.findall(r"https?://[^\s)\]>\"']+", polished))
    use_polished = (
        bool(polished)
        and all(heading in polished for heading in required_headings)
        and polished_urls == seen
    )
    report = (polished if use_polished else draft).strip() + "\n"
    path = ctx.run_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return str(path)
