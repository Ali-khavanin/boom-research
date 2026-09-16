from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import httpx
import tomli_w
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from ragent.book_to_skill import build as build_skill
from ragent.config import DEFAULT_CONFIG, Config, load_config, require_api_key
from ragent.errors import GraphError, RagentError
from ragent.executor.runner import run
from ragent.graph_builder import Graph, Metric, audit, build_graph, load_seed, to_mermaid, write_mermaid
from ragent.llm import get_ledger, get_llm
from ragent.trajectories import load_runs, to_csv
from ragent.wiki_refiner import gated, load_proposal, merge, propose

app = typer.Typer(help="Research Skill-Graph Agent")
providers_app = typer.Typer(help="Inspect configured LLM providers")
graph_app = typer.Typer(help="Build, view, and audit research graphs")
runs_app = typer.Typer(help="Inspect and export research runs")
refine_app = typer.Typer(help="Offline graph refinement")
app.add_typer(providers_app, name="providers")
app.add_typer(graph_app, name="graph")
app.add_typer(runs_app, name="runs")
app.add_typer(refine_app, name="refine")
console = Console()


@dataclass(slots=True)
class State:
    cfg: Config
    config_path: Path | None


def _state(ctx: typer.Context) -> State:
    value = ctx.ensure_object(dict).get("state")
    if not isinstance(value, State):
        raise RuntimeError("CLI state was not initialized")
    return value


def _graph(path: Path) -> Graph:
    if not path.exists():
        raise RagentError(f"graph not found: {path}")
    try:
        return Graph.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise GraphError(f"invalid graph {path}: {exc}") from exc


def _print_audit(graph: Graph) -> bool:
    result = audit(graph)
    console.print(f"entry={result.entry}")
    console.print(f"done reachable from {result.done_reachable_count}/{result.node_count} nodes")
    console.print("dead ends: " + (", ".join(result.dead_ends) if result.dead_ends else "none"))
    for finding in result.findings:
        style = "red" if finding.level == "error" else "green"
        console.print(f"[{style}]{finding.level}: {finding.detail}[/{style}]")
    return result.ok


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="TOML configuration path")] = None,
    workspace: Annotated[Path | None, typer.Option("--workspace", help="Workspace override")] = None,
    model: Annotated[list[str] | None, typer.Option("--model", help="Repeatable role=provider/model override")] = None,
) -> None:
    try:
        cfg = load_config(config, workspace=workspace, model_overrides=model)
    except (ValueError, ValidationError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    ctx.ensure_object(dict)["state"] = State(cfg=cfg, config_path=config)


@app.command("init")
def init_command(ctx: typer.Context) -> None:
    state = _state(ctx)
    config_path = state.config_path or Path("ragent.toml")
    if not config_path.exists():
        config_path.write_text(tomli_w.dumps(DEFAULT_CONFIG), encoding="utf-8")
        console.print(f"created {config_path}")
    env_path = Path(".env.example")
    if not env_path.exists():
        env_path.write_text(
            "OPENROUTER_API_KEY=\nOPENAI_API_KEY=\nGOOGLE_API_KEY=\nTAVILY_API_KEY=\n",
            encoding="utf-8",
        )
        console.print(f"created {env_path}")
    for relative in ("skills", "runs", "graph_versions", "refine"):
        (state.cfg.workspace / relative).mkdir(parents=True, exist_ok=True)
    console.print(f"workspace ready: {state.cfg.workspace}")


def _ping_provider(name: str, state: State) -> tuple[bool, str]:
    provider = state.cfg.providers[name]
    key = require_api_key(name, provider)
    headers = {"Authorization": f"Bearer {key}"}
    if provider.kind == "google":
        url = (provider.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/") + "/models"
        params = {"key": key}
        headers = {}
    else:
        default = "https://openrouter.ai/api/v1" if provider.kind == "openrouter" else "https://api.openai.com/v1"
        url = (provider.base_url or default).rstrip("/") + "/models"
        params = {}
    try:
        response = httpx.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        return True, f"HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


@providers_app.command("check")
def providers_check(ctx: typer.Context) -> None:
    state = _state(ctx)
    status: dict[str, tuple[bool, str]] = {}
    for name in state.cfg.providers:
        try:
            status[name] = _ping_provider(name, state)
        except RagentError as exc:
            status[name] = (False, str(exc))
    table = Table("Provider", "Kind", "Status")
    for name, provider in state.cfg.providers.items():
        ok, detail = status[name]
        table.add_row(name, provider.kind, ("[green]ok[/green] " if ok else "[red]failed[/red] ") + detail)
    console.print(table)
    roles = Table("Role", "Provider", "Model")
    for role, value in state.cfg.roles.items():
        roles.add_row(role, value.provider, value.model)
    console.print(roles)
    if not all(ok for ok, _ in status.values()):
        raise typer.Exit(1)


@app.command("book")
def book_command(
    ctx: typer.Context,
    pdf: Path,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    state = _state(ctx)
    if not pdf.exists():
        raise RagentError(f"PDF not found: {pdf}")
    target = out or state.cfg.workspace / "skills"

    def progress(event: dict) -> None:
        if event["event"] == "chapter_start":
            suffix = " (cached)" if event["cached"] else ""
            console.print(f"chapter {event['index']}/{event['total']}: {event['title']}{suffix}")
        elif event["event"] == "chapter_done":
            console.print(f"  → {event['path']}")
        elif event["event"] == "skill_start":
            console.print("synthesizing SKILL.md")

    bundle = build_skill(pdf, target, get_llm("book_to_skill", cfg=state.cfg), force=force, on_event=progress)
    console.print(f"skill: {bundle.skill_file}")
    console.print(f"chapters: {len(bundle.chapter_files)}")
    console.print(get_ledger(state.cfg).status_line())


@graph_app.command("build")
def graph_build(
    ctx: typer.Context,
    skill_dir: Annotated[Path | None, typer.Option("--skill-dir")] = None,
    out: Annotated[Path | None, typer.Option("--out")] = None,
    no_extend: Annotated[bool, typer.Option("--no-extend")] = False,
) -> None:
    state = _state(ctx)
    target = out or state.cfg.workspace / "graph.json"
    if no_extend:
        graph = load_seed()
    else:
        last_graph: dict | None = None

        def progress(event: dict) -> None:
            nonlocal last_graph
            if event["event"] == "graph_chapter_start":
                console.print(f"chapter {event['index']}/{event['total']} {event['chapter']}")
            elif event["event"] == "graph_delta":
                console.print(
                    f"+{len(event['new_nodes'])} nodes +{len(event['new_edges'])} edges "
                    f"(total {event['node_count']}/{event['edge_count']})"
                )
            elif event["event"] == "graph_complete":
                last_graph = event["graph"]

        try:
            graph = build_graph(
                skill_dir or state.cfg.workspace / "skills",
                get_llm("graph_builder", cfg=state.cfg),
                load_seed(),
                on_event=progress,
            )
        except GraphError as exc:
            if last_graph is not None:
                rejected = target.with_name("graph.rejected.json")
                rejected.parent.mkdir(parents=True, exist_ok=True)
                rejected.write_text(json.dumps(last_graph, indent=2), encoding="utf-8")
                console.print(f"[red]{exc}[/red]")
                console.print(f"rejected graph saved: {rejected}")
            raise
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"graph: {target}")
    write_mermaid(graph, target.with_suffix(".mmd"))
    console.print(f"mermaid: {target.with_suffix('.mmd')}")
    console.print(get_ledger(state.cfg).status_line())
    if not _print_audit(graph):
        raise typer.Exit(1)


@graph_app.command("show")
def graph_show(
    ctx: typer.Context,
    node: Annotated[str | None, typer.Option("--node")] = None,
    graph_path: Annotated[Path | None, typer.Option("--graph")] = None,
) -> None:
    state = _state(ctx)
    graph = _graph(graph_path or state.cfg.workspace / "graph.json")
    nodes = [graph.node(node)] if node else graph.nodes
    root = Tree(f"graph v{graph.version} entry={graph.entry}")
    for item in nodes:
        branch = root.add(f"[bold]{item.id}[/bold] — {item.title}" + (" [terminal]" if item.terminal else ""))
        for edge in graph.out_edges(item.id):
            branch.add(f"{edge.id} → {edge.target}  metric={edge.termination_metric.kind}:{edge.termination_metric.key}")
    console.print(root)


@graph_app.command("audit")
def graph_audit(
    ctx: typer.Context,
    graph_path: Annotated[Path | None, typer.Option("--graph")] = None,
) -> None:
    state = _state(ctx)
    if not _print_audit(_graph(graph_path or state.cfg.workspace / "graph.json")):
        raise typer.Exit(1)


@app.command("research")
def research_command(
    ctx: typer.Context,
    query: str,
    graph_path: Annotated[Path | None, typer.Option("--graph")] = None,
    start: Annotated[str | None, typer.Option("--start")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 24,
    report: Annotated[bool, typer.Option("--report/--no-report")] = True,
    obsidian: Annotated[bool, typer.Option("--obsidian/--no-obsidian")] = True,
) -> None:
    state = _state(ctx)
    graph = _graph(graph_path or state.cfg.workspace / "graph.json").model_copy(deep=True)
    for edge in graph.edges:
        if not obsidian:
            edge.tool_set = [name for name in edge.tool_set if name != "obsidian.note"]
        if not report and "report.generate" in edge.tool_set:
            edge.tool_set = [name for name in edge.tool_set if name != "report.generate"]
            edge.produces = "publication"
            edge.termination_metric = Metric(kind="artifact_exists", key="quick_test")
    def progress(event: dict) -> None:
        metric = event.get("metric")
        if metric:
            style = "green" if metric.get("passed") else "red"
            console.print(f"[{style}]{event.get('activity')} → {event.get('target')}: {metric.get('detail')}[/{style}]")
    result = run(graph, query, state.cfg, start=start, max_steps=max_steps, on_event=progress)
    console.print(f"run {result.run_id}: reached {result.final_node} in {result.steps} transitions")
    if result.report_path:
        console.print(f"report: {result.report_path}")
    console.print(get_ledger(state.cfg).status_line())


@runs_app.command("list")
def runs_list(ctx: typer.Context) -> None:
    runs = load_runs(_state(ctx).cfg.workspace)
    table = Table("Run", "Transitions", "Final target")
    for run_id, events in sorted(runs.items(), reverse=True):
        transitions = [event for event in events if "edge_id" in event]
        target = transitions[-1].get("target", "") if transitions else ""
        table.add_row(run_id, str(len(transitions)), str(target))
    console.print(table)


@runs_app.command("show")
def runs_show(ctx: typer.Context, run_id: str) -> None:
    state = _state(ctx)
    runs = load_runs(state.cfg.workspace)
    if run_id not in runs:
        raise RagentError(f"run not found: {run_id}")
    console.print_json(json.dumps(runs[run_id], ensure_ascii=False))
    report = state.cfg.workspace / "runs" / run_id / "report.md"
    if report.exists():
        console.print(f"report: {report}")


@runs_app.command("export")
def runs_export(ctx: typer.Context, csv_path: Path) -> None:
    console.print(f"exported: {to_csv(_state(ctx).cfg.workspace, csv_path)}")


@refine_app.command("propose")
def refine_propose(ctx: typer.Context) -> None:
    proposal = propose(_state(ctx).cfg)
    console.print(f"proposal: {proposal.written_path}")
    for operation in proposal.ops:
        console.print(f"{operation.id}: {operation.op} — {operation.rationale}")


@refine_app.command("merge")
def refine_merge(
    ctx: typer.Context,
    proposal_path: Path,
    accept: Annotated[list[str] | None, typer.Option("--accept")] = None,
    all_ops: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    selected = ["all"] if all_ops else (accept or [])
    console.print(f"merged graph: {merge(load_proposal(proposal_path), selected, _state(ctx).cfg)}")


@refine_app.command("rollback")
def refine_rollback(ctx: typer.Context, candidate: Path) -> None:
    console.print_json(json.dumps(gated(_state(ctx).cfg, candidate), indent=2))


@app.command("tui")
def tui_command(ctx: typer.Context) -> None:
    from .tui import run_tui

    run_tui(_state(ctx).cfg)


def entrypoint() -> int:
    try:
        app()
    except RagentError as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
