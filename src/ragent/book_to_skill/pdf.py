from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from book_to_skill import ExtractionError, extract_single_file

from ragent.errors import RagentError


@dataclass(slots=True)
class ExtractedBook:
    text: str
    pages: int
    metadata: dict[str, Any]


def extract(pdf: Path, mode: str = "text") -> ExtractedBook:
    """Extract through the upstream book-to-skill engine.

    book-to-skill owns format detection, extractor fallbacks, sanitization,
    repeated header/footer cleanup, OCR detection, and structure metadata.
    """
    try:
        result = extract_single_file(
            pdf.expanduser().resolve(),
            extraction_mode=mode,
            install_mode="no",
        )
    except ExtractionError as exc:
        raise RagentError(str(exc)) from exc
    text = str(result.pop("text", "")).strip()
    if not text:
        raise RagentError(
            f"no extractable text in {pdf}; run OCR on the scanned PDF first"
        )
    return ExtractedBook(
        text=text,
        pages=int(result.get("pages") or 0),
        metadata=result,
    )
