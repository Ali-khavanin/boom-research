from __future__ import annotations

import json
import re
import selectors
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any
from ragent.llm import get_llm
from ragent.llm.base import Message

if TYPE_CHECKING:
    from ragent.executor.context import RunContext


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:96] or "note"


def _sources(ctx: "RunContext") -> str:
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
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
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
        preferred = next((name for name in names if "note" in name.lower() or "write" in name.lower()), None)
        if not preferred:
            return "Obsidian MCP server exposes no note/write tool"
        result = client.call(
            "tools/call",
            {"name": preferred, "arguments": {"title": title, "content": markdown}},
        )
        return json.dumps(result, ensure_ascii=False)
    finally:
        client.close()


def note(ctx: "RunContext", title: str, body: str, tags: list[str] | None = None) -> str:
    if ctx.cfg.obsidian.backend == "mcp" and not ctx.cfg.obsidian.mcp_command:
        return "Obsidian MCP backend configured without obsidian.mcp_command"
    vault = ctx.cfg.obsidian.vault_path
    if ctx.cfg.obsidian.backend == "vault" and vault is None:
        return "Obsidian vault backend configured without obsidian.vault_path"
    polished = get_llm("obsidian", cfg=ctx.cfg).text(
        [
            Message(
                "system",
                "Edit this Obsidian note body for clarity and useful internal links. "
                "Preserve every fact and URL; never add evidence. Return only Markdown without front matter.",
            ),
            Message("user", body),
        ]
    ).strip()
    source_rows = _sources(ctx)
    front = "---\n" + f"tags: {json.dumps(tags or ['research'])}\n" + "---\n\n"
    markdown = front + (polished or body.strip()) + "\n"
    if source_rows:
        markdown += "\n## Sources\n\n" + source_rows + "\n"
    if ctx.cfg.obsidian.backend == "mcp":
        try:
            return _mcp_note(ctx.cfg.obsidian.mcp_command, title, markdown)
        except Exception as exc:
            return f"Obsidian MCP error: {exc}"
    assert vault is not None
    folder = Path(vault).expanduser() / ctx.cfg.obsidian.folder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_slug(title)}.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)
