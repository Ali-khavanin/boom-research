from __future__ import annotations

import html
import os
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
import trafilatura

from ragent.errors import ProviderError

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
            self.results.append({"title": html.unescape("".join(self._text).strip()), "url": url})
            self._in_result = False
            self._href = None


def search(ctx: "RunContext", query: str, k: int | None = None) -> list[dict[str, Any]]:
    limit = min(k or ctx.cfg.search.max_results, ctx.cfg.search.max_results)
    with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
        if ctx.cfg.search.backend == "tavily":
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
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
                for item in response.json().get("results", [])[:limit]
            ]
        response = client.get(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; ragent/0.1)"},
        )
        response.raise_for_status()
    parser = _DDGParser()
    parser.feed(response.text)
    return parser.results[:limit]


def fetch(ctx: "RunContext", url: str) -> dict[str, str]:
    with httpx.Client(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
        response = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ragent/0.1)"})
        response.raise_for_status()
    extracted = trafilatura.extract(
        response.text,
        include_links=True,
        include_formatting=True,
        output_format="markdown",
    ) or ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.I | re.S)
    title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip()) if title_match else url
    citation = {"url": str(response.url), "title": title}
    if citation["url"] not in {item.get("url") for item in ctx.citations}:
        ctx.citations.append(citation)
    return {"url": citation["url"], "title": title, "text": extracted[:12_000]}
