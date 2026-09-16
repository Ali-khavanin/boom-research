from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .pdf import ExtractedBook

_CHAPTER = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+)?"
    r"((?:chapter|unit|lesson|module|lecture|part|section)\s+"
    r"(?:\d{1,3}|[IVXLCDM]{1,7})\b[^\n]*)"
)

_NUMBERED = re.compile(r"(?m)^[ \t]*(\d{1,2})[.)]?[ \t]+([A-Z][^\n]{2,80}?)[ \t]*$")


def _numbered_sections(text: str) -> list[tuple[int, int, str]]:
    """Longest ascending 1,2,3… run of numbered headings: (number, start_offset, title)."""
    candidates: list[tuple[int, int, str]] = []
    for match in _NUMBERED.finditer(text):
        number = int(match.group(1))
        if number > 20:
            continue
        candidates.append((number, match.start(), match.group(2).strip()))
    counts: dict[str, int] = {}
    for _, _, title in candidates:
        normalized = " ".join(title.lower().split())
        counts[normalized] = counts.get(normalized, 0) + 1
    filtered = [
        candidate
        for candidate in candidates
        if counts[" ".join(candidate[2].lower().split())] == 1
    ]
    runs: list[list[tuple[int, int, str]]] = []
    for candidate in filtered:
        if runs and candidate[0] == runs[-1][-1][0] + 1:
            runs[-1].append(candidate)
        else:
            runs.append([candidate])
    qualifying = [run for run in runs if run[0][0] == 1 and len(run) >= 3]
    if not qualifying:
        return []
    qualifying.sort(key=lambda run: run[0][1])
    return qualifying[-1]


@dataclass(slots=True)
class Chapter:
    index: int
    title: str
    text: str
    pages: list[int]


def _page_range(book: ExtractedBook, start: int, end: int) -> list[int]:
    if book.pages <= 0 or not book.text:
        return []
    first = max(1, math.floor(start / len(book.text) * book.pages) + 1)
    last = min(book.pages, max(first, math.ceil(end / len(book.text) * book.pages)))
    return list(range(first, last + 1))


def segment(book: ExtractedBook) -> list[Chapter]:
    """Segment upstream-cleaned text while honoring its structure metadata."""
    matches = list(_CHAPTER.finditer(book.text))
    detected = int(book.metadata.get("chapters_detected") or 0)
    if len(matches) >= 2:
        chapters: list[Chapter] = []
        if matches[0].start() > 500:
            chapters.append(
                Chapter(
                    index=1,
                    title="Front Matter",
                    text=book.text[: matches[0].start()].strip(),
                    pages=_page_range(book, 0, matches[0].start()),
                )
            )
        for position, match in enumerate(matches):
            end = matches[position + 1].start() if position + 1 < len(matches) else len(book.text)
            text = book.text[match.start() : end].strip()
            if text:
                chapters.append(
                    Chapter(
                        index=len(chapters) + 1,
                        title=match.group(1).strip(),
                        text=text,
                        pages=_page_range(book, match.start(), end),
                    )
                )
        return chapters
    sections = _numbered_sections(book.text)
    if sections:
        chapters = []
        if sections[0][1] > 500:
            chapters.append(
                Chapter(
                    index=1,
                    title="Front Matter",
                    text=book.text[: sections[0][1]].strip(),
                    pages=_page_range(book, 0, sections[0][1]),
                )
            )
        for position, (number, start, title) in enumerate(sections):
            end = sections[position + 1][1] if position + 1 < len(sections) else len(book.text)
            text = book.text[start:end].strip()
            if text:
                chapters.append(
                    Chapter(
                        index=len(chapters) + 1,
                        title=f"{number} {title}",
                        text=text,
                        pages=_page_range(book, start, end),
                    )
                )
        return chapters
    block_count = detected or max(1, math.ceil((book.pages or 12) / 12))
    block_size = max(1, math.ceil(len(book.text) / block_count))
    chapters = []
    for start in range(0, len(book.text), block_size):
        end = min(len(book.text), start + block_size)
        index = len(chapters) + 1
        chapters.append(
            Chapter(
                index=index,
                title=f"Section {index}",
                text=book.text[start:end].strip(),
                pages=_page_range(book, start, end),
            )
        )
    return chapters
