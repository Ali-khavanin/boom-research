from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from ragent.errors import ExportError
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
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]")
_FENCE = re.compile(r"(?ms)^(`{3,}|~{3,}).*?^\1\s*$")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:96] or "note"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(ctx: RunContext) -> str:
    rows: list[str] = []
    seen: set[str] = set()
    for citation in ctx.citations:
        url = citation.get("url", "")
        if url and url not in seen:
            seen.add(url)
            rows.append(f"- [{citation.get('title') or url}]({url})")
    return "\n".join(rows)


class _MCPClient:
    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.next_id = 1

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self.next_id
        self.next_id += 1
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
            + "\n"
        )
        self.process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=30):
            raise TimeoutError(f"MCP request timed out: {method}")
        response = json.loads(self.process.stdout.readline())
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        return dict(response.get("result") or {})

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()


def _mcp_note(command: list[str], title: str, markdown: str) -> str:
    client = _MCPClient(command)
    try:
        client.call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ragent", "version": "0.1.0"},
            },
        )
        listed = client.call("tools/list", {})
        tools = listed.get("tools") or []
        names = [tool.get("name", "") for tool in tools]
        preferred = next(
            (
                name
                for name in names
                if "note" in name.lower() or "write" in name.lower()
            ),
            None,
        )
        if not preferred:
            raise ExportError("Obsidian MCP server exposes no note/write tool")
        result = client.call(
            "tools/call",
            {"name": preferred, "arguments": {"title": title, "content": markdown}},
        )
        return json.dumps(result, ensure_ascii=False)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        raise ExportError(f"Obsidian MCP export failed: {exc}") from exc
    finally:
        client.close()


def note(
    ctx: RunContext, title: str, body: str, tags: list[str] | None = None
) -> str:
    if ctx.cfg.obsidian.backend == "mcp" and not ctx.cfg.obsidian.mcp_command:
        raise ExportError("Obsidian MCP backend requires obsidian.mcp_command")
    vault = ctx.cfg.obsidian.vault_path
    if ctx.cfg.obsidian.backend == "vault" and vault is None:
        raise ExportError("Obsidian vault backend requires obsidian.vault_path")
    polished = (
        get_llm("obsidian", cfg=ctx.cfg)
        .text(
            [
                Message(
                    "system",
                    "Edit this Obsidian note body for clarity. Preserve every fact and URL; never add "
                    "evidence or internal links. Return only Markdown without front matter.",
                ),
                Message("user", body),
            ]
        )
        .strip()
    )
    source_rows = _sources(ctx)
    front = "---\n" + f"tags: {json.dumps(tags or ['research'])}\n" + "---\n\n"
    markdown = front + (polished or body.strip()) + "\n"
    if source_rows:
        markdown += "\n## Sources\n\n" + source_rows + "\n"
    if ctx.cfg.obsidian.backend == "mcp":
        return _mcp_note(ctx.cfg.obsidian.mcp_command, title, markdown)
    assert vault is not None
    root = vault.expanduser().resolve(strict=True)
    if not (root / ".obsidian").is_dir():
        raise ExportError(f"configured path is not an Obsidian vault: {root}")
    folder = _safe_folder(root, ctx.cfg.obsidian.folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not folder.resolve(strict=True).is_relative_to(root):
        raise ExportError("Obsidian folder escapes the configured vault")
    path = folder / f"{_slug(title)}.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)


def _safe_folder(vault: Path, configured: str) -> Path:
    relative = Path(configured)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExportError("Obsidian folder must be relative and cannot contain '..'")
    target = vault.joinpath(*relative.parts)
    existing = target
    while not existing.exists() and existing != vault:
        existing = existing.parent
    if not existing.resolve(strict=True).is_relative_to(vault):
        raise ExportError("Obsidian folder escapes the vault through a symlink")
    return target


def _tags(configured: list[str], kind: str) -> list[str]:
    result: list[str] = []
    for raw in [*configured, kind]:
        value = re.sub(r"\s+", "-", raw.lstrip("#").strip().casefold())
        if not value or not re.fullmatch(r"[\w/-]+", value, re.UNICODE):
            raise ExportError(f"invalid Obsidian tag: {raw!r}")
        if value not in result:
            result.append(value)
    return result


def _frontmatter(
    *,
    title: str,
    tags: list[str],
    run_id: str,
    kind: str,
    source_url: str | None = None,
) -> str:
    fields: list[tuple[str, Any]] = [
        ("title", _visible_wikilinks(title, set())),
        ("tags", tags),
        ("run_id", run_id),
        ("type", kind),
    ]
    if source_url is not None:
        fields.append(("source_url", source_url))
    return (
        "---\n"
        + "\n".join(
            f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in fields
        )
        + "\n---\n\n"
    )


def _visible_wikilinks(text: str, allowed: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip().removesuffix(".md")
        alias = (match.group(2) or target.rsplit("/", 1)[-1]).strip()
        if target in allowed:
            return f"[[{target}|{alias}]]"
        return alias

    pieces: list[str] = []
    cursor = 0
    for fenced in _FENCE.finditer(text):
        pieces.append(_WIKILINK.sub(replace, text[cursor : fenced.start()]))
        pieces.append(fenced.group(0))
        cursor = fenced.end()
    pieces.append(_WIKILINK.sub(replace, text[cursor:]))
    return "".join(pieces)


def _outgoing(text: str) -> list[str]:
    unfenced = _FENCE.sub("", text)
    return sorted(
        {
            match.group(1).strip().removesuffix(".md")
            for match in _WIKILINK.finditer(unfenced)
        }
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_bundle(ctx: RunContext) -> str:
    if ctx.cfg.obsidian.backend != "vault" or ctx.cfg.obsidian.layout != "bundle":
        raise ExportError(
            "bundle export requires obsidian.backend='vault' and layout='bundle'"
        )
    if ctx.cfg.obsidian.vault_path is None:
        raise ExportError("bundle export requires obsidian.vault_path")
    vault = ctx.cfg.obsidian.vault_path.expanduser().resolve(strict=True)
    if not vault.is_dir() or not (vault / ".obsidian").is_dir():
        raise ExportError(f"configured path is not an existing Obsidian vault: {vault}")
    folder = _safe_folder(vault, ctx.cfg.obsidian.folder)
    folder.mkdir(parents=True, exist_ok=True)
    folder = folder.resolve(strict=True)
    if not folder.is_relative_to(vault):
        raise ExportError("Obsidian destination escapes the configured vault")
    destination = folder / ctx.run_id
    staging = folder / f".{ctx.run_id}.staging-{uuid4().hex}"
    related_hashes: dict[str, tuple[Path, str]] = {}
    related_targets: set[str] = set()
    for configured in ctx.cfg.obsidian.related_notes:
        relative = Path(configured)
        if relative.is_absolute() or ".." in relative.parts:
            raise ExportError(f"unsafe related note path: {configured}")
        relative_md = (
            relative if relative.suffix == ".md" else Path(str(relative) + ".md")
        )
        path = (vault / relative_md).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(vault):
            raise ExportError(
                f"related note is missing or outside the vault: {configured}"
            )
        target = path.relative_to(vault).with_suffix("").as_posix()
        related_targets.add(target)
        related_hashes[target] = (path, _hash(path))
    report_path = Path(ctx.artifacts.get("report_path", ""))
    if not report_path.is_file() or not report_path.read_text(encoding="utf-8").strip():
        raise ExportError("verified report file is required before bundle export")
    missing = [key for _, key in _SECTIONS if not ctx.artifacts.get(key, "").strip()]
    if missing:
        raise ExportError("bundle export is missing stages: " + ", ".join(missing))
    source_by_url = {
        normalize_url(item.get("url", "")): item
        for item in ctx.citations
        if normalize_url(item.get("url", ""))
    }
    if len(source_by_url) < ctx.cfg.research.min_sources:
        raise ExportError(
            f"bundle has {len(source_by_url)} fetched sources; requires {ctx.cfg.research.min_sources}"
        )
    prefix = destination.relative_to(vault).as_posix()
    index_target = f"{prefix}/index"
    report_target = f"{prefix}/report"
    stage_targets = {key: f"{prefix}/stages/{key}" for _, key in _SECTIONS}
    source_targets = {
        url: f"{prefix}/sources/{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        for url in source_by_url
    }
    generated_targets = {
        index_target,
        report_target,
        *stage_targets.values(),
        *source_targets.values(),
    }
    allowed = generated_targets | related_targets
    citations_by_note: dict[str, set[str]] = {}
    report_text = report_path.read_text(encoding="utf-8")
    citations_by_note[report_target] = cited_source_urls(report_text, ctx)
    for _, key in _SECTIONS:
        citations_by_note[stage_targets[key]] = cited_source_urls(
            ctx.artifacts[key], ctx
        )
    try:
        staging.mkdir(parents=False, exist_ok=False)
        base_tags = ctx.cfg.obsidian.tags
        child_links: list[tuple[str, str]] = [(report_target, "Report")]
        child_links.extend((stage_targets[key], heading) for heading, key in _SECTIONS)
        child_links.extend(
            (source_targets[url], str(source_by_url[url].get("title") or url))
            for url in sorted(source_targets)
        )
        related_links = "\n".join(
            f"- [[{target}|{target.rsplit('/', 1)[-1]}]]"
            for target in sorted(related_targets)
        )
        navigation = "\n".join(
            f"- [[{target}|{alias}]]" for target, alias in child_links
        )
        index = _frontmatter(
            title=ctx.query,
            tags=_tags(base_tags, "index"),
            run_id=ctx.run_id,
            kind="index",
        )
        index += (
            f"# {ctx.query}\n\n"
            "## Review method\n\n"
            + (
                ctx.cfg.research.instructions.strip()
                or "Narrative review of fetched evidence."
            )
            + "\n\n## Navigation\n\n"
            + navigation
            + ("\n\n## Related notes\n\n" + related_links if related_links else "")
            + "\n"
        )
        _write(staging / "index.md", _visible_wikilinks(index, allowed))

        def evidence_links(text: str) -> str:
            urls = cited_source_urls(text, ctx)
            if not urls:
                return ""
            return "\n\n## Evidence\n\n" + "\n".join(
                f"- [[{source_targets[url]}|{source_by_url[url].get('title') or url}]]"
                for url in sorted(urls)
            )

        report_note = _frontmatter(
            title=f"Report — {ctx.query}",
            tags=_tags(base_tags, "report"),
            run_id=ctx.run_id,
            kind="report",
        )
        report_note += (
            f"[[{index_target}|Back to index]]\n\n"
            + report_text.strip()
            + evidence_links(report_text)
            + "\n"
        )
        _write(staging / "report.md", _visible_wikilinks(report_note, allowed))
        for heading, key in _SECTIONS:
            artifact = ctx.artifacts[key]
            stage = _frontmatter(
                title=f"{heading} — {ctx.query}",
                tags=_tags(base_tags, "stage"),
                run_id=ctx.run_id,
                kind="stage",
            )
            stage += (
                f"[[{index_target}|Back to index]]\n\n# {heading}\n\n"
                + artifact.strip()
                + evidence_links(artifact)
                + "\n"
            )
            _write(staging / "stages" / f"{key}.md", _visible_wikilinks(stage, allowed))
        for url, citation in sorted(source_by_url.items()):
            evidence = ctx.run_dir / citation["evidence_path"]
            if not evidence.is_file():
                raise ExportError(f"source evidence is missing: {evidence}")
            excerpt = evidence.read_text(encoding="utf-8")
            if hashlib.sha256(excerpt.encode()).hexdigest() != citation.get(
                "text_sha256"
            ):
                raise ExportError(f"source evidence hash mismatch: {url}")
            backlinks = sorted(
                target for target, urls in citations_by_note.items() if url in urls
            )
            fence = "`" * max(
                3, max((len(run) for run in re.findall(r"`+", excerpt)), default=0) + 1
            )
            title = str(citation.get("title") or url)
            source = _frontmatter(
                title=title,
                tags=_tags(base_tags, "source"),
                run_id=ctx.run_id,
                kind="source",
                source_url=url,
            )
            source += (
                f"[[{index_target}|Back to index]]\n\n"
                f"# {_visible_wikilinks(title, set())}\n\n"
                f"- Original URL: {citation.get('original_url', url)}\n"
                f"- Final URL: {url}\n"
                f"- Retrieved: {citation.get('fetched_at', '')}\n\n"
                "## Retrieved excerpt\n\n"
                "This is the exact retrieved excerpt retained by the research run, not a complete paper or model summary.\n\n"
                f"{fence}text\n{excerpt}\n{fence}\n"
            )
            if backlinks:
                source += (
                    "\n## Cited by\n\n"
                    + "\n".join(
                        f"- [[{target}|{target.rsplit('/', 1)[-1]}]]"
                        for target in backlinks
                    )
                    + "\n"
                )
            filename = source_targets[url].rsplit("/", 1)[-1] + ".md"
            _write(staging / "sources" / filename, source)
        notes: dict[str, dict[str, Any]] = {}
        for note_path in sorted(staging.rglob("*.md")):
            relative = note_path.relative_to(staging).as_posix()
            text = note_path.read_text(encoding="utf-8")
            outgoing = _outgoing(text)
            invalid = sorted(set(outgoing) - allowed)
            if invalid:
                raise ExportError(
                    f"generated note {relative} has unresolved wikilinks: {', '.join(invalid)}"
                )
            if relative == "index.md":
                kind = "index"
            elif relative == "report.md":
                kind = "report"
            elif relative.startswith("stages/"):
                kind = "stage"
            else:
                kind = "source"
            notes[relative] = {
                "sha256": _hash(note_path),
                "tags": _tags(base_tags, kind),
                "outgoing_wikilinks": outgoing,
            }
        manifest = {
            "version": 1,
            "run_id": ctx.run_id,
            "query": ctx.query,
            "vault_relative_root": prefix,
            "notes": notes,
            "tags": _tags(base_tags, "research"),
            "related_notes": {
                target: digest for target, (_, digest) in related_hashes.items()
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if destination.exists():
            existing_manifest = destination / "manifest.json"
            if not existing_manifest.is_file():
                raise ExportError(
                    f"bundle destination already exists without a manifest: {destination}"
                )
            try:
                existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ExportError(
                    f"existing bundle manifest is unreadable: {exc}"
                ) from exc
            if existing != manifest:
                raise ExportError("same-run bundle conflicts with existing manifest")
            for relative, metadata in notes.items():
                existing_note = destination / relative
                if (
                    not existing_note.is_file()
                    or _hash(existing_note) != metadata["sha256"]
                ):
                    raise ExportError(
                        f"same-run bundle note was modified or removed: {relative}"
                    )
            shutil.rmtree(staging)
        else:
            os.replace(staging, destination)
        for target, (related_path, digest) in related_hashes.items():
            if _hash(related_path) != digest:
                raise ExportError(f"related note changed during publication: {target}")
        actual_manifest = destination / "manifest.json"
        ctx.artifacts["obsidian_manifest_path"] = str(actual_manifest)
        ctx.artifacts["obsidian_index_path"] = str(destination / "index.md")
        return str(actual_manifest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
