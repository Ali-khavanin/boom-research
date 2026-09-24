# Trajectories

## Definition

A trajectory is the append-only event log of one admitted call to `run()`: the file
`<workspace>/runs/<run_id>/trace.jsonl`, one JSON object per line. New runs pair it with authoritative
`run.json` metadata and offline `verification.json`; transition-only CSV compatibility is preserved.

`run_id` is constructed in `_make_context` (`executor/runner.py`) as:

```python
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
```

i.e. a UTC timestamp such as `20260916T120000Z` followed by `-` and the first 8 hex characters of a random UUID4, e.g. `20260916T120000Z-3f9a1c2d`.

CLI inspection rejects run IDs containing path traversal. Provider/graph/skill/vault preflight occurs
before `_make_context`, so admission failure creates no run directory. Once admitted,
`runs/<run_id>/artifacts/` is created and version-1 `run.json` is written immediately with:

`version`, `run_id`, `query`, `status`, `final_node`, `requirements`, `models`, `graph`, `skill`,
`pricing`, `budget`, `obsidian`, and `outputs`.

`status` is `running`, `completed`, or `failed`. The effective graph is copied to `graph.json`; an
explicit skill is copied verbatim to `skill.md`; both metadata entries record relative path and
SHA-256. Requirements record minimum sources and required report/bundle outputs. Budget metadata
records strict mode, campaign limit, absolute ledger path, and before/after snapshots. Output keys are
`report_path`, `obsidian_manifest_path`, and `obsidian_index_path`, using null until produced.

Unlike historical traces, `run.json` records the research query and resolved execution contract.
Historical directories without `run.json` remain readable as traces but `runs verify` reports that
provenance is unavailable rather than synthesizing it.

## Record shape

`TraceLog.append` (`src/ragent/trajectories/log.py`) is the single write path; it prepends three fields to every record before writing:

```python
event = {
    "case_id": self.run_id,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "step": self.step,
    **event,
}
```

`case_id` is the run id, `timestamp` is an ISO-8601 UTC timestamp taken at write time, and `step` is a 0-based counter private to that `TraceLog` instance, incremented after every `append` regardless of record kind (transition or error) — so `step` is a strict per-run sequence number across both record kinds.

**Transition records** come from `TraceLog.transition`, called once per stage attempt from the executor's main loop in `runner.py`:

```python
ctx.trace.transition(
    node_id=node.id,
    edge_id=edge.id,
    target=edge.target,
    metric={
        "kind": edge.termination_metric.kind,
        "passed": metric.passed,
        "detail": metric.detail,
    },
    artifact_key=edge.produces,
    artifact_chars=len(artifact),
    tool_calls=tool_calls,
    model=llm.label,
    attempt=attempt,
    tokens=after["total_tokens"] - before["total_tokens"],
    cost_usd=round(after["cost_usd"] - before["cost_usd"], 6),
)
```

which `append`s a dict with exactly these keys beyond the common three: `activity` (set from `node_id`, i.e. the *source* node of the attempted edge), `edge_id`, `target` (`edge.target`), `metric` (`{kind, passed, detail}`), `artifact_key` (`edge.produces`), `artifact_chars` (`len(artifact)` — Python string length, so **characters**, not bytes), `tool_calls` (list of tool names actually invoked during the attempt), `model` (`llm.label`, the resolved provider/model string for that step), `attempt`, `tokens`, `cost_usd`.

- **`target` is written unconditionally, on every attempt, whether the metric passed or failed.** The runner computes `edge.target` and passes it into `transition` before it knows or branches on `metric.passed` — a failed attempt whose control flow actually stays on the same edge (retry) or jumps to `edge.on_fail` still logs `target: edge.target`, the edge's *nominal* destination, not the node the run actually continues from. Reading `target` off a failed-metric record does not tell you where the run went next; only a *passing* record's `target` is where `node` is reassigned to next.
- `attempt` is prior failures plus one: the runner computes `attempt = ctx.attempts[edge.id] + 1` (`ctx.attempts` is a `Counter[str]` on `RunContext`, incremented only after a failure) before logging, then increments `ctx.attempts[edge.id]` afterward only if the metric failed. So the first try of an edge logs `attempt: 1`, and `ctx.attempts` accumulates across every revisit of that edge id within the run.
- `tokens` and `cost_usd` are session-ledger deltas covering the entire attempt, including judge
  usage because the post-attempt snapshot is taken after metric evaluation.

**Usage records** are sanitized ledger events subscribed during the run. They identify role/model,
request tokens/cost, session/lifetime totals, and strict reservation/exposure fields. They have no
`edge_id`, so CSV export continues to exclude them.

**Search records** use `event=\"search\"` and record query, backend, result count, and returned URLs.

**Completion records** use `event=\"complete\"` and are appended only after required output checks,
with final report/index paths. `run.json` is then persisted as completed and `verify_run` checks both
the passed terminal transition and this completion record.

**Error records** use `event=\"error\"`, source node in `activity`, and `detail`. Failures update
`run.json` to `failed`; a late error therefore overrides an earlier nominal terminal target for new
run summaries.

### Process-mining framing

The record shape maps directly onto standard process-mining vocabulary: `case_id` is the case (one research run), `activity` is the event class (the source node a stage was attempted from), and `timestamp`/`step` give two independent, agreeing total orderings of events within a case (wall-clock and log-sequence). This is exactly the `case_id, activity, timestamp` triple that process-mining tools (event-log miners, conformance checkers) expect as their minimal input, which is why `to_csv` below exists as a dedicated narrow export rather than shipping the full JSONL to those tools directly.

## Reading and exporting

`load_runs(workspace: Path) -> dict[str, list[dict[str, Any]]]` (`src/ragent/trajectories/store.py`):

```python
def load_runs(workspace: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    run_root = workspace / "runs"
    if not run_root.exists():
        return runs
    for trace_path in sorted(run_root.glob("*/trace.jsonl")):
        events: list[dict[str, Any]] = []
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        runs[trace_path.parent.name] = events
    return runs
```

It globs `runs/*/trace.jsonl` in sorted (lexical, hence chronological given the `run_id` timestamp prefix) order, keys the result by each trace file's parent directory name, skips blank lines, and performs **no shape validation whatsoever** — every non-blank line is handed to `json.loads` and appended as-is. A malformed JSON line propagates a `json.JSONDecodeError` straight out of `load_runs` (and therefore out of every caller: `runs list`, `runs show`, `runs export`, `stats`, `propose`); there is no per-line try/except. An empty or missing `runs/` directory yields `{}` rather than an error.

`to_csv(workspace: Path, path: Path) -> Path` (same file):

```python
CSV_FIELDS = ("case_id", "activity", "timestamp", "step", "edge_id", "metric_passed")
```

`to_csv` writes a header row from that exact `CSV_FIELDS` tuple, then iterates every run's events in `load_runs` order and, for each event, **skips it entirely if `"edge_id" not in event`** — which excludes every error record (error records never carry `edge_id`) from the CSV. Surviving rows are written as `case_id`, `activity`, `timestamp`, `step`, `edge_id`, and `metric_passed` (`event.get("metric", {}).get("passed", False)`, defaulting to `False` if no `metric` key is present). Everything else present in the raw JSONL — `target`, the metric's `kind`/`detail`, `artifact_key`, `artifact_chars`, `tool_calls`, `model`, `attempt`, `tokens`, `cost_usd` — is deliberately dropped from the CSV; it remains available only by reading `trace.jsonl` directly (via `load_runs` or `ragent runs show`). The CSV is therefore a minimal process-mining event log, not a full trajectory export.

CLI surface (`src/ragent/cli/main.py`, `runs_app`):

- `ragent runs list` prefers explicit `run.json.status` for new runs and falls back to the historical
  last-transition target only when metadata is absent.
- `ragent runs show RUN_ID` containment-checks the ID and prints `run.json`, trace events, and output
  paths.
- `ragent runs verify RUN_ID` is offline. It rechecks the saved graph/skill/evidence hashes, six
  required stages, fetched-and-cited source threshold, report references, bundle frontmatter/tags,
  every wikilink and note hash, related-note hashes, effective model labels, strict campaign exposure
  with zero reservations, completed state, passed terminal transition, and completion record. It
  writes `verification.json` and exits nonzero on named failures.
- `ragent runs export CSV_PATH` retains transition-only CSV output.

## Aggregation

`stats(workspace: Path) -> dict[str, Any]` (`src/ragent/trajectories/store.py`) is the sole aggregation function, and it is also the only input `propose` sends to the refiner LLM (see below). Its exact return shape:

```python
{"edges": per_edge, "incomplete_runs": incomplete, "failures": failures[:12]}
```

- **`edges`** — `dict[edge_id, {"attempts": int, "pass_rate": float, "mean_artifact_size": float}]`, one entry per edge id that appears in at least one transition event across all runs, sorted by edge id. `attempts` is `len(events)` for that edge — a raw **count of transition events** referencing that edge id (across every run and every retry), not a distinct-run count and not a max of any `attempt` field. `pass_rate` is `passed / len(events)` where `passed` sums `bool(event["metric"]["passed"])` over those same events, i.e. passing-event fraction, not a per-run success rate. `mean_artifact_size` is `statistics.mean` of each event's `artifact_chars` (defaulting a missing key to `0` via `event.get("artifact_chars", 0)`), so it is in characters, matching the trace field's own unit.
- **`incomplete_runs`** — new runs are complete only when `run.json.status == "completed"`; this
  prevents a late error from being hidden by an earlier passed nominal `done` transition. Historical
  runs without metadata retain the old passed-terminal inference.
- **`failures`** — one `{"run_id", "edge_id", "detail"}` dict per transition event whose `metric["passed"]` is falsy, built while iterating runs in `load_runs` order (and events in file order within each run), then **hard-capped to the first twelve** via `failures[:12]`. `incomplete_runs` and every per-edge count in `edges` are **not** capped — only the `failures` list is truncated.

## Offline refinement gate

Refinement lives in `src/ragent/wiki_refiner/` — `proposer.py`, `maintainer.py`, `evalset.py`, `rollback.py` — re-exported from `src/ragent/wiki_refiner/__init__.py` as `Proposal`, `ProposalOp`, `propose`, `load_proposal`, `merge`, `gated`. (An earlier note referred to this as `graph_builder/refine.py`; that module does not exist — the real package is `wiki_refiner`, and the function names `propose`/`merge`/`gated` below are the actual ones, verified by reading the source, not guesses.)

### `propose`

`propose(cfg: Config) -> Proposal` (`wiki_refiner/proposer.py`) sends the refiner LLM **only** `stats(cfg.workspace)` — never `graph.json`, raw artifacts, the query text, or full `trace.jsonl` contents:

```python
evidence = stats(cfg.workspace)
llm = get_llm("refiner", cfg=cfg)
schema = Proposal.model_json_schema()
raw = llm.json([...], schema)
```

System message, verbatim:

```
Propose minimal graph maintenance from process-mining evidence. Never delete stages. Use unique snake_case operation ids. Prefer fixing a repeatedly failing edge or metric over adding nodes. Payloads must be complete for add operations and targeted for updates.
```

User message, verbatim prefix followed by the JSON-dumped `stats()` evidence:

```
Trajectory statistics and up to twelve concrete failure details follow. Return one or more evidence-backed operations.
```

The raw JSON reply is validated as `Proposal.model_validate(raw)`, then written to `<workspace>/refine/<%Y%m%dT%H%M%SZ>/proposal.json` (UTC timestamp directory, via `proposal.model_dump_json(indent=2)`), and `proposal.written_path` is set to that path before the `Proposal` is returned. `ragent refine propose` prints `proposal: <written_path>` followed by one `<op.id>: <op.op> — <op.rationale>` line per operation.

`Proposal` is `{"ops": list[ProposalOp]}` with `ops` required to be non-empty (`Field(min_length=1)`). `ProposalOp` is `{"id": str, "op": Literal["add_edge", "update_edge", "update_metric", "add_node"], "payload": dict, "rationale": str}` — **there is no delete operation**; only these four literal values are accepted by the schema, and a fifth value would fail Pydantic validation before `propose` ever returns.

Per-op payload validation, enforced by a `model_validator` on `ProposalOp` at construction time:

- `add_edge` — `payload` must validate as a full `Edge` (`Edge.model_validate(self.payload)`); no custom error text, failures surface as the underlying Pydantic `ValidationError`.
- `add_node` — `payload` must validate as a full `Node` (`Node.model_validate(self.payload)`), same failure mode.
- `update_metric` — `payload` must contain both `edge_id` and `metric`, else exactly:
  ```
  update_metric requires edge_id and metric
  ```
  and `payload["metric"]` must validate as a `Metric` (`Metric.model_validate(...)`).
- `update_edge` — `payload` must contain a truthy `id` and a dict-typed `changes`, else exactly:
  ```
  update_edge requires id and changes
  ```

### `merge`

`merge(proposal: Proposal, accept: list[str], cfg: Config | None = None) -> Path` (`wiki_refiner/maintainer.py`), exposed as `ragent refine merge PROPOSAL [--accept ID]... [--all]`:

- Loads `<workspace>/graph.json`, raising `GraphError(f"current graph not found: {graph_path}")` if it is missing.
- Selection: if the literal string `"all"` is anywhere in `accept`, every op in the proposal is selected; otherwise the selection is `[op for op in proposal.ops if op.id in set(accept)]`. The CLI builds `accept` as `["all"]` when `--all` is passed, else the list of `--accept` values (or `[]`). An empty selection raises exactly:
  ```
  no proposal operations were selected
  ```
- Before applying any operation, `merge` snapshots the *current* graph — unconditionally, even if the merge later fails — to `<workspace>/graph_versions/<%Y%m%dT%H%M%SZ>/graph.json` via `shutil.copy2(graph_path, version_dir / "graph.json")`.
- Then, per selected op, in list order:
  - `add_node` — appends a validated `Node`; duplicate id raises `add_node id already exists: {node.id}`.
  - `add_edge` — appends a validated `Edge`; duplicate id raises `add_edge id already exists: {edge.id}`.
  - `update_metric` — looks up `graph.edges` by `payload["edge_id"]`; missing target raises `update_metric edge not found: {edge_id}`; on success, replaces that edge's `termination_metric` with a freshly validated `Metric`.
  - `update_edge` — looks up `graph.edges` by `payload["id"]`; missing target raises `update_edge edge not found: {edge_id}`; on success, dumps the existing edge to a dict, applies `payload["changes"]` as an overlay (`dict.update`), and revalidates the merged dict as a new `Edge`.
- After all selected ops are applied, `merge` runs `audit(graph)` (from `graph_builder.validate`) on the mutated in-memory graph. If the audit is not clean, it raises:
  ```
  proposal regresses graph audit: {details}
  ```
  where `details` joins every `error`-level finding's `detail` with `"; "`. Only when the audit passes does `merge` overwrite `graph.json` in place (`graph.model_dump_json(indent=2)`) and return that path. There is no separate merge audit log file — the only persisted record of the attempt is the pre-merge snapshot under `graph_versions/` (which survives regardless of whether the merge ultimately succeeds or raises).

### `gated`

`gated(cfg: Config, candidate_graph: Path | Graph) -> dict[str, Any]` (`wiki_refiner/rollback.py`) is what the CLI exposes, somewhat confusingly, as `ragent refine rollback CANDIDATE` — the command name refers to the rollback-on-rejection behavior inside `gated`, not a separate rollback function; there is no other rollback entry point.

Precondition: the candidate (loaded from a `Graph` object or a JSON file path) must itself pass `audit`, else:
```
candidate graph failed audit: {details}
```
(same `"; "`-joined error-finding detail format as `merge`'s audit-failure message).

Both the current graph (`<workspace>/graph.json`, required to exist — `current graph not found: {current_path}` if not) and the candidate are then scored by `_score(graph, cfg)`, which runs `executor.runner.run(graph, query, cfg, on_event=events.append)` once per query in `load_eval_set()` and computes a `Scorecard(success, quality, queries)`:

- `success = successes / count` — `successes` counts queries for which `run()` returned without raising and `result.reached_done` was true (an exception during a query's run is caught and simply does not increment `successes`); `count` is the number of eval queries.
- `quality = first_passes / metric_events if metric_events else 0.0` — over every logged transition event across every eval query, `metric_events` counts events carrying a `metric` dict at all, and `first_passes` counts the subset where `event["attempt"] == 1` **and** `metric["passed"]` is true. So `quality` is the fraction of *first-attempt* stage transitions that passed their metric, out of all stage-transition attempts (first or retried).

Acceptance: `kept = candidate.success >= baseline.success and candidate.quality >= baseline.quality - 0.02` — the candidate must not regress `success` at all, and may regress `quality` by at most `0.02`.

- If `kept`, `graph.json` is overwritten with the candidate.
- If not kept, `graph.json` is restored from the lexically latest `graph_versions/*/graph.json` snapshot (`sorted(...)[−1]`), if one exists; if no snapshot exists, `graph.json` is left untouched.

Every call to `gated` (accepted or rejected) writes `<workspace>/refine/<%Y%m%dT%H%M%SZ>/rollback.json` shaped:

```json
{
  "kept": true,
  "baseline": {"success": 0.0, "quality": 0.0, "queries": 3},
  "candidate": {"success": 0.0, "quality": 0.0, "queries": 3},
  "threshold": {"success_drop": 0.0, "quality_drop": 0.02}
}
```

(`baseline`/`candidate` are `dataclasses.asdict(Scorecard(...))`; `threshold` is the fixed pair of constants used in the acceptance check above, not derived from config.) The same dict is returned to the caller and, from the CLI, printed as JSON.

The three fixed evaluation queries, `src/ragent/data/eval_set.json`, verbatim:

```json
[
  "retrieval-augmented generation for clinical question answering",
  "methods for detecting distribution shift in deployed classifiers",
  "privacy-preserving synthetic data for rare-disease research"
]
```

loaded via `load_eval_set()` in `wiki_refiner/evalset.py` (`json.loads(files("ragent.data").joinpath("eval_set.json").read_text(...))`).

### Refinement never runs during `ragent research`

`research_command` in `src/ragent/cli/main.py` imports only `run` from `ragent.executor.runner`; it never imports or calls `propose`, `merge`, or `gated`, and nothing in `executor/runner.py` references `wiki_refiner`. Refinement is exclusively a separate, explicit, offline step (`ragent refine propose` / `ragent refine merge` / `ragent refine rollback`) run by an operator against accumulated trajectories after the fact.

`gated` scoring is not free: `_score` calls the real `run()` executor once per eval query for both the baseline and the candidate graph — four `run()` invocations minimum (2 graphs × the 3 fixed queries, though the constant is `len(load_eval_set())` per side) — each of which drives the full tool-and-LLM loop described in [algorithm](algorithm.md) and spends real tokens (and, on OpenRouter, real billed cost) exactly like an ordinary `ragent research` call.

## See also

- [algorithm](algorithm.md) — the `run()` executor loop that produces the transition and error records described here.
- [skill-graph](skill-graph.md) — the graph schema (`Node`, `Edge`, `Metric`) that `add_node`/`add_edge`/`update_metric`/`update_edge` payloads validate against, and the `audit` findings that gate `merge` and `gated`.
- [README](README.md) — documentation index.
