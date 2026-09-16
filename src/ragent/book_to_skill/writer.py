from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ragent.errors import RagentError
from ragent.llm.base import LLM, Message

from .chapters import Chapter, segment
from .pdf import extract

_CUES = ("first", "then", "before", "if", "unless", "in order to", "so that", "until")


@dataclass(slots=True)
class SkillBundle:
    root: Path
    skill_file: Path
    chapter_files: list[Path]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:72] or "section"


def _truncate(text: str) -> str:
    if len(text) <= 24_000:
        return text
    return text[:16_000] + "\n\n[...middle omitted...]\n\n" + text[-8_000:]


def _stage_hints(chapter: Chapter) -> list[str]:
    lowered = chapter.text.lower()
    return [cue for cue in _CUES if cue in lowered]


def _chapter_prompt(chapter: Chapter) -> list[Message]:
    return [
        Message(
            "system",
            "Convert research-methodology source text into an executable skill chapter. "
            "Do not invent methods or evidence.",
        ),
        Message(
            "user",
            "Write Markdown with these sections: Purpose, When to use, Ordered procedure, "
            "Decision rules, and Discourse cues. Preserve relevant cue words verbatim, "
            "especially first, then, before, if, unless, in order to, so that, and until. "
            f"Source title: {chapter.title}\nSource pages: {chapter.pages}\n\n{_truncate(chapter.text)}",
        ),
    ]


def build(
    pdf: Path,
    out_dir: Path,
    llm: LLM,
    force: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> SkillBundle:
    def _emit(payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(payload)
        except Exception:
            pass

    chapters = segment(extract(pdf))
    if not chapters:
        raise RagentError(f"no chapters could be segmented from {pdf}")
    out_dir.mkdir(parents=True, exist_ok=True)
    chapter_dir = out_dir / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    rendered: list[str] = []
    _emit({"event": "book_start", "pdf": str(pdf), "chapters": len(chapters)})
    for chapter in chapters:
        path = chapter_dir / f"{chapter.index:02d}-{_slug(chapter.title)}.md"
        paths.append(path)
        cached = path.exists() and not force
        _emit(
            {
                "event": "chapter_start",
                "index": chapter.index,
                "total": len(chapters),
                "title": chapter.title,
                "chars": len(chapter.text),
                "cached": cached,
            }
        )
        if cached:
            body = path.read_text(encoding="utf-8")
        else:
            content = llm.text(_chapter_prompt(chapter)).strip()
            front = (
                "---\n"
                f"name: {json.dumps(chapter.title)}\n"
                f"source_pages: {json.dumps(chapter.pages)}\n"
                f"stage_hints: {json.dumps(_stage_hints(chapter))}\n"
                "---\n\n"
            )
            body = front + content + "\n"
            path.write_text(body, encoding="utf-8")
        rendered.append(body)
        _emit(
            {
                "event": "chapter_done",
                "index": chapter.index,
                "total": len(chapters),
                "title": chapter.title,
                "path": str(path),
                "cached": cached,
            }
        )
    summary_prompt = [
        Message(
            "system",
            "Synthesize chapter skills into one faithful end-to-end research procedure. "
            "Keep stage order, decision rules, loops, and source chapter links explicit.",
        ),
        Message(
            "user",
            "Produce the body of SKILL.md in Markdown. Include Overview, End-to-end procedure, "
            "Decision points, and Chapters. Do not add YAML front matter.\n\n"
            + "\n\n".join(rendered),
        ),
    ]
    skill_path = out_dir / "SKILL.md"
    skill_cached = skill_path.exists() and not force
    _emit({"event": "skill_start", "cached": skill_cached})
    if not skill_cached:
        summary = llm.text(summary_prompt).strip()
        front = (
            "---\n"
            f"name: {json.dumps(pdf.stem)}\n"
            f"description: {json.dumps('Research methodology extracted from ' + pdf.name)}\n"
            f"chapters: {json.dumps([str(path.relative_to(out_dir)) for path in paths])}\n"
            "---\n\n"
        )
        skill_path.write_text(front + summary + "\n", encoding="utf-8")
    _emit({"event": "skill_done", "path": str(skill_path)})
    return SkillBundle(root=out_dir, skill_file=skill_path, chapter_files=paths)
