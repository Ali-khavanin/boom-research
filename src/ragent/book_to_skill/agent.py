from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ragent.config import BookCfg
from ragent.errors import RagentError

from .skill import SkillArtifacts, load_skill

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def skill_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:64].strip("-")
    if not slug:
        raise RagentError(f"cannot derive a skill name from {value!r}; pass --name")
    return slug


def _prompt(source: Path, slug: str, skills_root: Path, mode: str, depth: str) -> str:
    content_type = "1 (Technical)" if mode == "technical" else "2 (Text-heavy)"
    purpose = (
        "3 (Reference specific chapters and concepts)"
        if depth == "reference"
        else "4 (All of the above)"
    )
    return f'''/book-to-skill "{source}" {slug}

This is a non-interactive Full Conversion started by ragent; nobody can answer questions, so use these answers:
- Step 1.5 content type: {content_type}, so BOOK_TYPE={mode}.
- Step 2.5 cost estimate: proceed with Full Conversion.
- Step 4 purpose: {purpose}, so DEPTH={depth}.
- Step 5 destination: I explicitly request SKILLS_HOME="{skills_root}". Write the skill to "{skills_root}/{slug}/" exactly: no category subfolder, no symlink, no `hermes skills trust`, and nothing under ~/.hermes/skills or ~/.agents/skills.
- Missing optional extractor packages: do not install them; use the available fallback.
- Step 11 publish: skip.'''


def generate_skill(
    source: Path,
    skills_root: Path,
    cfg: BookCfg,
    *,
    name: str | None = None,
    mode: str = "text",
    depth: str = "study",
    force: bool = False,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> SkillArtifacts:
    def _emit(payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(payload)
        except Exception:
            pass

    source = source.expanduser().resolve()
    if not source.is_file():
        raise RagentError(f"source not found: {source}")
    if mode not in ("text", "technical"):
        raise RagentError(f"mode must be text or technical: {mode}")
    if depth not in ("study", "reference"):
        raise RagentError(f"depth must be study or reference: {depth}")
    slug = name or skill_slug(source.stem)
    if _SLUG.fullmatch(slug) is None or len(slug) > 64:
        raise RagentError(f"invalid skill name: {slug!r}")
    skills_root = skills_root.expanduser().resolve()
    target = skills_root / slug
    if target.exists() and not force:
        artifacts = load_skill(target)
        _emit(
            {
                "event": "skill_done",
                "path": str(artifacts.skill_file),
                "chapters": len(artifacts.chapter_files),
                "cached": True,
            }
        )
        return artifacts
    if target.exists() and force:
        shutil.rmtree(target)
    skills_root.mkdir(parents=True, exist_ok=True)
    argv = [
        part.replace("{prompt}", _prompt(source, slug, skills_root, mode, depth))
        for part in cfg.agent
    ]
    if shutil.which(argv[0]) is None:
        raise RagentError(
            f"book-to-skill host agent not found on PATH: {argv[0]} "
            "(set [book].agent in ragent.toml)"
        )
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "BOOK_SKILL_INSTALL_MISSING": "no",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    _emit(
        {
            "event": "book_start",
            "pdf": str(source),
            "skill_dir": str(target),
            "agent": argv[0],
        }
    )
    with subprocess.Popen(
        argv,
        cwd=skills_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            _emit({"event": "agent_output", "line": line.rstrip("\n")})
        code = proc.wait()
    if code != 0:
        raise RagentError(f"book-to-skill agent exited with status {code}")
    artifacts = load_skill(target)
    _emit(
        {
            "event": "skill_done",
            "path": str(artifacts.skill_file),
            "chapters": len(artifacts.chapter_files),
            "cached": False,
        }
    )
    return artifacts
