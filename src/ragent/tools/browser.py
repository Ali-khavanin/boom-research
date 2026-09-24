from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
import trafilatura
from pypdf import PdfReader

from ragent.errors import ProviderError
from ragent.executor.metrics import normalize_url

if TYPE_CHECKING:
    from ragent.executor.context import RunContext


class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []
        self._in_result = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "a" and "result__a" in (values.get("class") or ""):
            self._href = values.get("href")
            self._text = []
            self._in_result = True

    def handle_data(self, data: str) -> None:
        if self._in_result:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result and self._href:
            url = self._href
            query = parse_qs(urlparse(url).query)
            if "uddg" in query:
                url = unquote(query["uddg"][0])
            self.results.append(
                {"title": html.unescape("".join(self._text).strip()), "url": url}
            )
            self._in_result = False
            self._href = None


def _trace_search(
    ctx: RunContext, query: str, backend: str, results: list[dict[str, Any]]
) -> None:
    ctx.trace.append(
        {
            "event": "search",
            "query": query,
            "backend": backend,
            "result_count": len(results),
            "urls": [str(item.get("url", "")) for item in results],
        }
    )


def search(ctx: RunContext, query: str, k: int | None = None) -> list[dict[str, Any]]:
    limit = min(k or ctx.cfg.search.max_results, ctx.cfg.search.max_results)
    backend = ctx.cfg.search.backend
    with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
        if backend == "tavily":
            key = os.getenv(ctx.cfg.search.api_key_env)
            if not key:
                raise ProviderError(
                    f"Tavily search requires environment variable {ctx.cfg.search.api_key_env}"
                )
            response = client.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": limit},
            )
            response.raise_for_status()
            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
                for item in response.json().get("results", [])[:limit]
            ]
            _trace_search(ctx, query, backend, results)
            return results
        response = client.get(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; ragent/0.1)"},
        )
        response.raise_for_status()
    lowered = response.text.lower()
    if any(
        marker in lowered for marker in ("captcha", "anomaly-modal", "challenge-form")
    ):
        _trace_search(ctx, query, backend, [])
        raise ProviderError("DuckDuckGo returned a challenge instead of search results")
    parser = _DDGParser()
    parser.feed(response.text)
    results = parser.results[:limit]
    _trace_search(ctx, query, backend, results)
    if not results:
        raise ProviderError("DuckDuckGo returned no parseable search results")
    return results


def _extract(response: httpx.Response) -> tuple[str, str]:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    url = str(response.url)
    if content_type == "application/pdf" or urlparse(url).path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(response.content))
            text = "\n\n".join(
                (page.extract_text() or "") for page in reader.pages
            ).strip()
        except Exception as exc:
            raise ProviderError(f"PDF evidence is unreadable: {exc}") from exc
        if not text:
            raise ProviderError("PDF evidence has no extractable text")
        return text[:12_000], url
    if content_type.startswith("text/plain") or content_type in {
        "text/markdown",
        "text/x-markdown",
    }:
        text = response.text.strip()
        if not text:
            raise ProviderError("text evidence is empty")
        return text[:12_000], url
    if (
        content_type
        and "html" not in content_type
        and not content_type.startswith("text/")
    ):
        raise ProviderError(f"unsupported evidence content type: {content_type}")
    extracted = (
        trafilatura.extract(
            response.text,
            include_links=True,
            include_formatting=True,
            output_format="markdown",
        )
        or ""
    ).strip()
    if not extracted:
        raise ProviderError("HTML evidence extraction returned empty text")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.IGNORECASE | re.DOTALL)
    title = (
        html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip())
        if title_match
        else url
    )
    return extracted[:12_000], title


def _write_index(ctx: RunContext) -> None:
    payload = {
        "version": 1,
        "sources": ctx.citations,
        "aliases": ctx.source_aliases,
    }
    path = ctx.run_dir / "sources.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch(ctx: RunContext, url: str) -> dict[str, str]:
    requested = normalize_url(url)
    if not requested:
        raise ProviderError("browser.fetch accepts only absolute HTTP(S) URLs")
    with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
        response = client.get(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; ragent/0.1)"}
        )
        response.raise_for_status()
    canonical = normalize_url(str(response.url))
    if not canonical:
        raise ProviderError("fetch redirect resolved outside HTTP(S)")
    existing = next(
        (
            item
            for item in ctx.citations
            if normalize_url(item.get("url", "")) == canonical
        ),
        None,
    )
    ctx.source_aliases[requested] = canonical
    if existing is not None:
        _write_index(ctx)
        evidence_path = ctx.run_dir / existing["evidence_path"]
        return {
            "url": existing["url"],
            "title": existing["title"],
            "text": evidence_path.read_text(encoding="utf-8"),
        }
    text, title = _extract(response)
    sources_dir = ctx.run_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16] + ".md"
    evidence_path = sources_dir / filename
    if evidence_path.exists():
        raise ProviderError(f"source evidence collision: {evidence_path}")
    evidence_path.write_text(text, encoding="utf-8")
    citation = {
        "url": canonical,
        "original_url": requested,
        "title": title,
        "fetched_at": datetime.now(UTC).isoformat(),
        "evidence_path": evidence_path.relative_to(ctx.run_dir).as_posix(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    try:
        proposed = [*ctx.citations, citation]
        payload = {
            "version": 1,
            "sources": proposed,
            "aliases": ctx.source_aliases,
        }
        (ctx.run_dir / "sources.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        evidence_path.unlink(missing_ok=True)
        raise
    ctx.citations.append(citation)
    return {"url": canonical, "title": title, "text": text}
