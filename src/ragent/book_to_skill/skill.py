from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ragent.errors import RagentError

SUPPORTING_FILES = ("glossary.md", "patterns.md", "cheatsheet.md")
_CHAPTER = re.compile(r"^ch(\d+)-.+\.md$")


@dataclass(slots=True)
class SkillArtifacts:
    root: Path
    skill_file: Path
    chapter_files: list[Path]
    supporting_files: list[Path]


def load_skill(root: Path) -> SkillArtifacts:
    chapter_dir = root / "chapters"
    chapters: list[tuple[int, str, Path]] = []
    if chapter_dir.is_dir():
        for path in chapter_dir.iterdir():
            match = _CHAPTER.fullmatch(path.name)
            if match is not None and path.is_file():
                chapters.append((int(match.group(1)), path.name, path))
    chapter_files = [path for _, _, path in sorted(chapters)]
    missing = [
        name for name in ("SKILL.md", *SUPPORTING_FILES) if not (root / name).is_file()
    ]
    if not chapter_files:
        missing.append("chapters/ch<NN>-<slug>.md")
    if missing:
        raise RagentError(
            f"incomplete book-to-skill output in {root}: missing {', '.join(missing)}"
        )
    return SkillArtifacts(
        root=root,
        skill_file=root / "SKILL.md",
        chapter_files=chapter_files,
        supporting_files=[root / name for name in SUPPORTING_FILES],
    )


def find_skill(skills_root: Path) -> Path:
    candidates = sorted(p.parent for p in skills_root.glob("*/SKILL.md"))
    if not candidates:
        raise RagentError(
            f"no generated skill under {skills_root}; run `ragent book <pdf>` or pass --skill-dir"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RagentError(
            f"multiple skills under {skills_root}: {names}; pass --skill-dir"
        )
    return candidates[0]
