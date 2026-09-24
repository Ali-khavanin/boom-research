from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Markdown,
    OptionList,
    RichLog,
    Static,
    Tree,
)

from ragent.book_to_skill import build as build_skill
from ragent.config import Config
from ragent.errors import BudgetError
from ragent.executor.runner import run
from ragent.graph_builder import (
    Edge,
    Graph,
    Node,
    audit,
    build_graph,
    load_seed,
    to_mermaid,
    write_mermaid,
)
from ragent.llm import get_llm
from ragent.llm.usage import ModelPrice, UsageLedger, get_ledger


class TransitionMsg(Message):
    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event


class ArtifactMsg(Message):
    def __init__(self, key: str, text: str) -> None:
        super().__init__()
        self.key = key
        self.text = text


class UsageMsg(Message):
    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event


class PipelineMsg(Message):
    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__()
        self.event = event


@dataclass(frozen=True, slots=True)
class SlashCommand:
    keyword: str
    usage: str
    summary: str


_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/book", "/book <pdf-path>", "run the full book→graph pipeline"),
    SlashCommand("/budget", "/budget", "show usage totals and the per-model breakdown"),
    SlashCommand("/budget tokens", "/budget tokens N", "set the session token limit"),
    SlashCommand("/budget cost", "/budget cost USD", "set the session cost limit"),
    SlashCommand(
        "/budget price",
        "/budget price provider/model IN OUT",
        "set a manual per-mtok price",
    ),
    SlashCommand("/budget reset", "/budget reset", "reset session totals"),
    SlashCommand("/budget off", "/budget off", "clear both budget limits"),
)


class CommandInput(Input):
    """Query input that lets the app intercept palette navigation keys before
    Input's inherited scroll bindings and the Screen's tab binding see them."""

    def on_key(self, event: events.Key) -> None:
        handler = getattr(self.app, "handle_palette_key", None)
        if handler is None:
            return
        if handler(event.key):
            event.stop()


def _edge_detail(edge: Edge, graph: Graph) -> str:
    metric = edge.termination_metric
    verification = f"`{metric.kind}` on artifact key `{metric.key}`"
    if metric.n is not None:
        verification += f", n={metric.n}"
    if metric.pattern is not None:
        verification += f", pattern=`{metric.pattern}`"
    if metric.rubric is not None:
        verification += f', rubric: "{metric.rubric}"'

    if edge.on_fail:
        failure = (
            f"routes back to `{edge.on_fail}` after {edge.max_attempts} failed attempts"
        )
    else:
        failure = f"retries this edge in place; hard error after {edge.max_attempts + 1} failures"

    tools = ", ".join(f"`{tool}`" for tool in edge.tool_set) or "none — prompt only"
    if edge.precondition is None:
        precondition = "none — always eligible"
    else:
        precondition = f"`{edge.precondition.kind}` on `{edge.precondition.key}`"

    lines = [
        f"# edge `{edge.id}`",
        "",
        f"`{edge.source}` → `{edge.target}`",
        "",
        "## Verification required to advance",
        verification,
        "",
        f"Failing it: {failure}",
        "",
        "## Context handed to the model",
        f"- **Tools bound:** {tools}",
        f"- **Precondition:** {precondition}",
        f"- **Produces artifact key:** `{edge.produces}`",
        f"- **Max attempts:** `{edge.max_attempts}`",
    ]
    if edge.provenance is not None:
        lines.append(
            f"- **Provenance:** chapter `{edge.provenance.get('chapter', '')}`, "
            f"cue `{edge.provenance.get('cue', '')}`"
        )
    lines.extend(
        [
            "",
            "### prompt_template",
            "```text",
            edge.prompt_template,
            "```",
        ]
    )
    return "\n".join(lines)


def _node_detail(node: Node, graph: Graph) -> str:
    lines = [
        f"# node `{node.id}`",
        "",
        f"## {node.title}",
        "",
        node.description or "_No description._",
        "",
        f"- **Terminal:** `{'true' if node.terminal else 'false'}`",
    ]
    if node.model is not None:
        fields = ", ".join(
            f"{name}=`{value}`"
            for name, value in node.model.model_dump(exclude_none=True).items()
        )
        lines.append(f"- **Model override:** {fields or 'present with no fields set'}")
    if node.provenance is not None:
        provenance = ", ".join(
            f"{name} `{value}`" for name, value in node.provenance.items()
        )
        lines.append(f"- **Provenance:** {provenance}")

    lines.extend(["", "## Outward edges"])
    outward = graph.out_edges(node.id)
    if outward:
        lines.extend(
            f"- `{edge.id}` → `{edge.target}` — gate `{edge.termination_metric.kind}:{edge.termination_metric.key}`"
            for edge in outward
        )
    else:
        lines.append("_No outward edges (dead end)._")
    return "\n".join(lines)


class ResearchApp(App[None]):
    CSS = """
    #budget { height: 1; padding: 0 1; background: $panel; }
    #top { height: 7; }
    #body { height: 1fr; }
    #graph { width: 34%; border: round $accent; }
    #right { width: 66%; }
    #log { height: 45%; border: round $accent; }
    #artifact { height: 55%; border: round $accent; padding: 1; }
    #breakdown { height: 40%; border: round $accent; }
    #palette { height: auto; max-height: 9; border: round $accent; }
    #models { width: 55%; }
    #model_override { width: 45%; }
    """
    BINDINGS = [
        ("r", "run", "Run"),
        ("b", "book", "Book"),
        ("g", "build_graph", "Build graph"),
        ("p", "pipeline", "Book→Graph"),
        ("a", "audit_graph", "Audit"),
        ("u", "budget", "Budget"),
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.graph: Graph | None = None
        self.node_rows: dict[str, Any] = {}
        self.edge_rows: dict[str, Any] = {}
        self.edge_labels: dict[str, str] = {}
        self.palette_matches: list[SlashCommand] = []
        self.ledger: UsageLedger | None = None
        self._usage_subscriber: Callable[[dict[str, Any]], None] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="budget")
        yield CommandInput(
            placeholder="Research query (or PDF path for Book)", id="query"
        )
        yield OptionList(id="palette")
        with Horizontal(id="top"):
            yield DataTable(id="models")
            yield Input(
                placeholder="role=provider/model, then Enter", id="model_override"
            )
        with Horizontal(id="body"):
            yield Tree("Research graph", id="graph")
            with Vertical(id="right"):
                yield RichLog(id="log", markup=True, wrap=True)
                yield Markdown("_Newest artifact appears here._", id="artifact")
                yield DataTable(id="breakdown")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#models", DataTable)
        table.add_columns("Role", "Provider", "Model")
        for role, value in self.cfg.roles.items():
            table.add_row(role, value.provider, value.model, key=role)
        query = self.query_one("#query", CommandInput)
        query.value = self.cfg.research.query or ""
        self.ledger = get_ledger(self.cfg)
        breakdown = self.query_one("#breakdown", DataTable)
        breakdown.add_columns("Model", "Calls", "In", "Out", "Total", "Cost")
        breakdown.display = False
        self.query_one("#palette", OptionList).display = False
        self._usage_subscriber = lambda event: self.post_message(UsageMsg(event))
        self.ledger.subscribe(self._usage_subscriber)
        self._refresh_budget()
        self._load_graph()

    def _load_graph(self) -> None:
        configured = self.cfg.research.graph_path
        path = configured or self.cfg.workspace / "graph.json"
        if configured is not None and not path.exists():
            raise FileNotFoundError(f"configured research graph not found: {path}")
        self.graph = (
            Graph.model_validate_json(path.read_text(encoding="utf-8"))
            if path.exists()
            else load_seed()
        )
        self._render_graph_tree()

    def _render_graph_tree(self) -> None:
        tree = self.query_one("#graph", Tree)
        tree.clear()
        self.node_rows.clear()
        self.edge_rows.clear()
        self.edge_labels.clear()
        for node in self.graph.nodes:
            branch = tree.root.add(f"{node.id} — {node.title}", data=node.id)
            self.node_rows[node.id] = branch
            for edge in self.graph.out_edges(node.id):
                metric = edge.termination_metric
                label = f"{edge.id} → {edge.target}  [{metric.kind}:{metric.key}]"
                self.edge_labels[edge.id] = label
                leaf = branch.add_leaf(Text(label), data=edge.id)
                self.edge_rows[edge.id] = leaf
        tree.root.expand()
        if self.graph.entry in self.node_rows:
            tree.select_node(self.node_rows[self.graph.entry])

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if not data or self.graph is None:
            return
        edge = next((edge for edge in self.graph.edges if edge.id == data), None)
        if edge is not None:
            detail = _edge_detail(edge, self.graph)
        else:
            node = next((node for node in self.graph.nodes if node.id == data), None)
            if node is None:
                self.query_one("#log", RichLog).write(
                    f"details available once the graph build completes: {data}"
                )
                return
            detail = _node_detail(node, self.graph)
        self.query_one("#artifact", Markdown).update(detail)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "query":
            self._refresh_palette(event.value)

    def _refresh_palette(self, value: str) -> None:
        typed = " ".join(value.split()).lower()
        if not typed.startswith("/"):
            self._hide_palette()
            return

        matches = [
            command for command in _COMMANDS if command.keyword.startswith(typed)
        ]
        if not matches:
            matches = sorted(
                (command for command in _COMMANDS if typed.startswith(command.keyword)),
                key=lambda command: len(command.keyword),
                reverse=True,
            )
        if not matches:
            self._hide_palette()
            return

        self.palette_matches = matches
        palette = self.query_one("#palette", OptionList)
        palette.clear_options()
        palette.add_options(
            f"{command.usage}  —  {command.summary}" for command in matches
        )
        palette.highlighted = 0
        palette.display = True

    def _hide_palette(self) -> None:
        self.palette_matches = []
        palette = self.query_one("#palette", OptionList)
        palette.display = False
        palette.clear_options()

    def handle_palette_key(self, key: str) -> bool:
        palette = self.query_one("#palette", OptionList)
        if not palette.display or not self.palette_matches:
            return False
        if key == "down":
            palette.highlighted = ((palette.highlighted or 0) + 1) % len(
                self.palette_matches
            )
            return True
        if key == "up":
            palette.highlighted = ((palette.highlighted or 0) - 1) % len(
                self.palette_matches
            )
            return True
        if key == "tab":
            self._complete_palette(palette.highlighted or 0)
            return True
        if key == "escape":
            self._hide_palette()
            return True
        return False

    def _complete_palette(self, index: int) -> None:
        if not 0 <= index < len(self.palette_matches):
            return
        command = self.palette_matches[index]
        value = command.keyword + (" " if command.usage != command.keyword else "")
        query = self.query_one("#query", CommandInput)
        query.value = value
        query.cursor_position = len(value)
        self._refresh_palette(value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "palette":
            return
        self._complete_palette(event.option_index)
        self.set_focus(self.query_one("#query", CommandInput))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query":
            raw = event.value.strip()
            if raw.startswith("/"):
                self._slash_command(raw)
                event.input.value = ""
            self._hide_palette()
            return
        if event.input.id != "model_override":
            return
        raw = event.value.strip()
        try:
            role, target = raw.split("=", 1)
            provider, model = target.split("/", 1)
            if role not in self.cfg.roles or provider not in self.cfg.providers:
                raise ValueError("unknown role or provider")
            self.cfg.roles[role].provider = provider
            self.cfg.roles[role].model = model
            table = self.query_one("#models", DataTable)
            table.update_cell(role, "Provider", provider)
            table.update_cell(role, "Model", model)
            event.input.value = ""
        except ValueError as exc:
            self.query_one("#log", RichLog).write(f"[red]invalid override:[/red] {exc}")

    def _event_callback(self, event: dict[str, Any]) -> None:
        self.post_message(TransitionMsg(event))
        key = event.get("artifact_key")
        case_id = event.get("case_id")
        if key and case_id:
            path = (
                self.cfg.workspace / "runs" / str(case_id) / "artifacts" / f"{key}.md"
            )
            if path.exists():
                self.post_message(
                    ArtifactMsg(str(key), path.read_text(encoding="utf-8"))
                )

    def on_transition_msg(self, message: TransitionMsg) -> None:
        event = message.event
        log = self.query_one("#log", RichLog)
        kind = event.get("event")
        if kind == "search":
            log.write(
                f"[cyan]search:[/cyan] {event.get('result_count', 0)} results "
                f"for {event.get('query', '')}"
            )
            return
        if kind == "usage":
            log.write(
                f"[blue]usage:[/blue] {event.get('role')} "
                f"${event.get('cost_usd') or 0:.6f}"
            )
            return
        if kind == "complete":
            log.write(
                f"[green]complete[/green] report={event.get('report_path') or 'disabled'} "
                f"index={event.get('obsidian_index_path') or 'disabled'}"
            )
            return
        metric = event.get("metric")
        if not isinstance(metric, dict):
            detail = event.get("detail", "")
            color = "red" if kind == "error" else "yellow"
            log.write(f"[{color}]{detail}[/{color}]")
            return
        passed = bool(metric.get("passed"))
        color = "green" if passed else "red"
        activity = str(event.get("activity", ""))
        target = str(event.get("target", ""))
        log.write(
            f"[{color}]{activity} → {target}: {metric.get('detail', '')}[/{color}]"
        )
        edge_id = str(event.get("edge_id", ""))
        if edge_id in self.edge_rows:
            base = self.edge_labels.get(edge_id, f"{edge_id} → {target}")
            self.edge_rows[edge_id].set_label(Text(base, style=color))
        if passed and target in self.node_rows:
            tree = self.query_one("#graph", Tree)
            tree.select_node(self.node_rows[target])

    def on_artifact_msg(self, message: ArtifactMsg) -> None:
        self.query_one("#artifact", Markdown).update(
            f"# {message.key}\n\n{message.text}"
        )

    def on_usage_msg(self, message: UsageMsg) -> None:
        self._refresh_budget()
        if message.event.get("warn") and self.ledger is not None:
            self.query_one("#log", RichLog).write(
                f"[red]budget warning: {self.ledger.status_line()}[/red]"
            )

    def on_pipeline_msg(self, message: PipelineMsg) -> None:
        event = message.event
        log = self.query_one("#log", RichLog)
        kind = event.get("event")
        if kind == "book_start":
            log.write(f"book: {event['chapters']} chapters from {event['pdf']}")
        elif kind == "chapter_start":
            suffix = " (cached)" if event.get("cached") else ""
            log.write(
                f"chapter {event['index']}/{event['total']}: {event['title']}{suffix}"
            )
        elif kind == "chapter_done":
            log.write(f"  wrote {event['path']}")
        elif kind == "skill_start":
            log.write(
                "SKILL.md: cached" if event.get("cached") else "SKILL.md: synthesizing"
            )
        elif kind == "skill_done":
            log.write(f"SKILL.md: {event['path']}")
        elif kind == "graph_start":
            self.graph = load_seed()
            self._render_graph_tree()
            log.write(f"graph: merging {event['chapters']} chapters")
        elif kind == "graph_delta":
            tree = self.query_one("#graph", Tree)
            for node in event["new_nodes"]:
                branch = tree.root.add(
                    f"[green]{node['id']} — {node['title']}[/green]", data=node["id"]
                )
                self.node_rows[node["id"]] = branch
            for edge in event["new_edges"]:
                parent = self.node_rows.get(edge["source"], tree.root)
                label = f"{edge['id']} → {edge['target']}"
                self.edge_labels[edge["id"]] = label
                leaf = parent.add_leaf(Text(label, style="green"), data=edge["id"])
                self.edge_rows[edge["id"]] = leaf
            tree.root.expand()
            log.write(
                f"{event['chapter']}: +{len(event['new_nodes'])} nodes +{len(event['new_edges'])} edges "
                f"→ {event['node_count']}/{event['edge_count']}"
            )
        elif kind == "graph_complete":
            graph = Graph.model_validate(event["graph"])
            self.graph = graph
            self._render_graph_tree()
            name = "graph.json" if event["audit_ok"] else "graph.rejected.json"
            path = self.cfg.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
            write_mermaid(graph, self.cfg.workspace / "graph.mmd")
            self.post_message(
                ArtifactMsg("graph.mmd", "```mermaid\n" + to_mermaid(graph) + "```")
            )
            color = "green" if event["audit_ok"] else "red"
            detail = "ok" if event["audit_ok"] else event["detail"]
            log.write(f"[{color}]audit: {detail}[/{color}]")
        elif kind == "pipeline_error":
            log.write(f"[red]{event['detail']}[/red]")

    def _refresh_budget(self) -> None:
        if self.ledger is None:
            return
        self.query_one("#budget", Static).update(self.ledger.status_line())
        breakdown = self.query_one("#breakdown", DataTable)
        if breakdown.display:
            breakdown.clear()
            by_model = self.ledger.snapshot()["by_model"]
            for model, totals in sorted(
                by_model.items(), key=lambda item: item[1]["total_tokens"], reverse=True
            ):
                breakdown.add_row(
                    model,
                    str(totals["calls"]),
                    f"{totals['prompt_tokens']:,}",
                    f"{totals['completion_tokens']:,}",
                    f"{totals['total_tokens']:,}",
                    f"${totals['cost_usd']:,.4f}",
                )

    _BUDGET_HELP = " | ".join(
        command.usage for command in _COMMANDS if command.usage.startswith("/budget")
    )

    def _slash_command(self, raw: str) -> None:
        log = self.query_one("#log", RichLog)
        parts = raw.split()
        if parts and parts[0] == "/book":
            if len(parts) < 2:
                log.write("[red]book: usage /book <pdf-path>[/red]")
                return
            path = Path(parts[1]).expanduser()
            if not path.exists():
                log.write(f"[red]book: file not found: {path}[/red]")
                return
            log.write(f"pipeline: {path}")
            self.run_pipeline(str(path))
            return
        if not parts or parts[0] != "/budget":
            log.write(self._BUDGET_HELP)
            log.write(_COMMANDS[0].usage)
            return
        if self.ledger is None:
            log.write("[red]budget: ledger not ready[/red]")
            return
        args = parts[1:]
        if not args:
            log.write(self.ledger.status_line())
            for model, totals in sorted(
                self.ledger.snapshot()["by_model"].items(),
                key=lambda item: item[1]["total_tokens"],
                reverse=True,
            ):
                log.write(
                    f"  {model}: {totals['calls']} calls, {totals['total_tokens']:,} tokens, "
                    f"${totals['cost_usd']:,.4f}"
                )
            breakdown = self.query_one("#breakdown", DataTable)
            breakdown.display = True
            self._refresh_budget()
            return
        try:
            sub = args[0]
            if sub == "tokens":
                value = int(args[1])
                self.ledger.set_limits(token_limit=value)
                self.cfg.budget.token_limit = value
                log.write(f"[green]budget: token limit set to {value:,}[/green]")
            elif sub == "cost":
                value = float(args[1])
                self.ledger.set_limits(cost_limit_usd=value)
                self.cfg.budget.cost_limit_usd = value
                log.write(f"[green]budget: cost limit set to ${value:,.2f}[/green]")
            elif sub == "off":
                self.ledger.set_limits(token_limit=None, cost_limit_usd=None)
                self.cfg.budget.token_limit = None
                self.cfg.budget.cost_limit_usd = None
                log.write("[green]budget: limits cleared[/green]")
            elif sub == "reset":
                self.ledger.reset_session()
                suffix = (
                    "; strict campaign exposure is unchanged"
                    if self.ledger.strict
                    else ""
                )
                log.write(f"[green]budget: session totals reset{suffix}[/green]")
            elif sub == "price":
                key = args[1]
                input_price = float(args[2])
                output_price = float(args[3])
                self.ledger.set_price(key, ModelPrice(input_price, output_price))
                log.write(f"[green]budget: price set for {key}[/green]")
            else:
                log.write(f"[red]budget: unknown subcommand '{sub}'[/red]")
                log.write(self._BUDGET_HELP)
                return
            self._refresh_budget()
        except (ValueError, IndexError, BudgetError) as exc:
            log.write(f"[red]budget: {exc}[/red]")
            log.write(self._BUDGET_HELP)

    def action_budget(self) -> None:
        breakdown = self.query_one("#breakdown", DataTable)
        breakdown.display = not breakdown.display
        self._refresh_budget()

    def on_unmount(self) -> None:
        if self.ledger is not None and self._usage_subscriber is not None:
            self.ledger.unsubscribe(self._usage_subscriber)

    @work(thread=True, exclusive=True)
    def action_run(self) -> None:
        query = self.query_one("#query", Input).value.strip()
        if query.startswith("/"):
            self.post_message(
                TransitionMsg({"detail": "slash commands run on Enter, not r"})
            )
            return
        if not query:
            self.post_message(TransitionMsg({"detail": "enter a research query"}))
            return
        graph = self.graph or load_seed()
        try:
            run(graph, query, self.cfg, on_event=self._event_callback)
        except Exception as exc:
            self.post_message(TransitionMsg({"detail": f"{type(exc).__name__}: {exc}"}))

    @work(thread=True, exclusive=True)
    def action_book(self) -> None:
        raw = self.query_one("#query", Input).value.strip()
        path = Path(raw).expanduser()
        if not path.exists():
            self.post_message(
                TransitionMsg(
                    {"detail": "Book binding uses the query field as a PDF path"}
                )
            )
            return
        try:
            bundle = build_skill(
                path,
                self.cfg.workspace / "skills",
                get_llm("book_to_skill", cfg=self.cfg),
            )
            self.post_message(
                ArtifactMsg("SKILL.md", bundle.skill_file.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            self.post_message(TransitionMsg({"detail": f"{type(exc).__name__}: {exc}"}))

    @work(thread=True, exclusive=True)
    def action_build_graph(self) -> None:
        try:
            graph = build_graph(
                self.cfg.workspace / "skills",
                get_llm("graph_builder", cfg=self.cfg),
                load_seed(),
            )
            path = self.cfg.workspace / "graph.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
            self.call_from_thread(self._load_graph)
            self.post_message(TransitionMsg({"detail": f"graph built: {path}"}))
        except Exception as exc:
            self.post_message(TransitionMsg({"detail": f"{type(exc).__name__}: {exc}"}))

    def action_pipeline(self) -> None:
        value = self.query_one("#query", Input).value.strip()
        if not value or value.startswith("/"):
            self.query_one("#log", RichLog).write(
                "pipeline: type /book <pdf-path> and press Enter"
            )
            return
        self.run_pipeline(value)

    @work(thread=True, exclusive=True)
    def run_pipeline(self, raw: str) -> None:
        pdf = Path(raw).expanduser()
        skills = self.cfg.workspace / "skills"
        emit = lambda event: self.post_message(PipelineMsg(event))
        try:
            bundle = build_skill(
                pdf, skills, get_llm("book_to_skill", cfg=self.cfg), on_event=emit
            )
            self.post_message(
                ArtifactMsg("SKILL.md", bundle.skill_file.read_text(encoding="utf-8"))
            )
            build_graph(
                skills,
                get_llm("graph_builder", cfg=self.cfg),
                load_seed(),
                on_event=emit,
            )
        except BudgetError as exc:
            emit({"event": "pipeline_error", "detail": f"budget stop: {exc}"})
        except Exception as exc:
            emit({"event": "pipeline_error", "detail": f"{type(exc).__name__}: {exc}"})

    def action_audit_graph(self) -> None:
        result = audit(self.graph or load_seed())
        color = "green" if result.ok else "red"
        details = "; ".join(item.detail for item in result.findings)
        self.query_one("#log", RichLog).write(f"[{color}]audit: {details}[/{color}]")


def run_tui(cfg: Config) -> None:
    ResearchApp(cfg).run()
