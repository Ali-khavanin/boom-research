# Research Skill-Graph Agent

`ragent` turns a research-methodology document into an executable, audited procedure:
upstream `book-to-skill` converts the PDF into a complete skill through a host agent,
then ragent compiles its chapters onto a seed research-stage graph. `ragent graph audit`
checks graph soundness (no dead ends, every node reaches `done`), and `ragent research`
executes queries by walking that graph one metric-gated transition at a time.
Every step is appended to a JSONL trajectory for export and offline graph refinement.

The MIT-licensed [`virgiliojr94/book-to-skill`](https://github.com/virgiliojr94/book-to-skill)
owns extraction, chapter selection, and skill generation. Its Python package supplies
extractor dependencies; generation is a host agent following upstream's `SKILL.md`,
not a Python generation API. ragent owns artifact validation, graph compilation,
the metric-gated executor, and trajectory/refinement tooling.

## Install

Requires Python >= 3.11 and Git.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

`pyproject.toml` pins `book-to-skill[pdf]` to Git commit
`c108d25b0cb58e1bdc361f3de02ed9f37075152f`. This unreleased revision adds Hermes
support missing from `v1.4.0`. GitHub access is required during installation.

**PDF conversion prerequisite:** install/configure Hermes, put `hermes` on `PATH`,
and install the upstream skill at the same revision:

```bash
git clone https://github.com/virgiliojr94/book-to-skill.git ~/.hermes/skills/research/book-to-skill
git -C ~/.hermes/skills/research/book-to-skill checkout --detach c108d25b0cb58e1bdc361f3de02ed9f37075152f
hermes skills list
```

The list must contain `book-to-skill`. See [the conversion guide](docs/book-to-skill.md)
for existing-clone updates and checking/upgrading a same-version Python installation.
Hermes uses its own configured model and credentials; its spend is not recorded or
limited by ragent's usage ledger.

Installing registers the `ragent` console script (`[project.scripts] ragent =
"ragent.cli.main:entrypoint"`). The distribution is `research-skill-agent` version `0.1.0`.

## Credentials

`ragent` never reads a `.env` file. Each provider's API key comes from a plain environment variable
named by that provider's `api_key_env`:

| Provider | Env var | Required when |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | any role routes through `openrouter` (the default for every role). |
| `openai` | `OPENAI_API_KEY` | a role is switched to `openai`. |
| `google` | `GOOGLE_API_KEY` | a role is switched to `google`. |
| Tavily search | `TAVILY_API_KEY` | `search.backend = "tavily"` (default backend is `ddg`, keyless). |

A missing key fails fast with the exact variable name: `provider '<name>' requires environment
variable <VAR>`. Providers never silently fall back to another provider or a cached response.

Practical note on OpenRouter keys: they are shaped `sk-or-v1-…`. Sending an OpenAI-format
`sk-proj-…` key to OpenRouter returns `401 Missing Authentication header`; sending a revoked or
unknown OpenRouter key returns `401 User not found`. Both are OpenRouter API responses, not `ragent`
errors — check `ragent providers check` output first.

## Quick start

```bash
ragent init                       # writes ragent.toml (if absent), .env.example, workspace dirs
ragent providers check            # pings every configured provider, prints role → provider/model
ragent book ./paper.pdf           # PDF -> .ragent/skills/paper/{SKILL.md,chapters/,glossary.md,patterns.md,cheatsheet.md}
ragent graph build                # the sole generated skill -> .ragent/graph.json
ragent research "your query"      # walks the graph, writes .ragent/runs/<run_id>/
ragent tui                        # three-pane interactive UI
```

`ragent graph build`'s streamed output looks like this — one line per chapter, then the compiled
graph, its Mermaid rendering, the usage ledger status line, and the audit result:

```
chapter 1/6 ch01-introduction.md
+2 nodes +2 edges (total 10/9)
chapter 2/6 ch02-literature-review.md
+0 nodes +0 edges (total 10/9)
...
graph: .ragent/graph.json
mermaid: .ragent/graph.mmd
tokens 41,208 (in 33,114 / out 8,094) · bill $0.0000 · session 12 calls · lifetime 41,208 tokens / $0.0000
entry=start
done reachable from 10/10 nodes
dead ends: none
info: graph reachability and dead-end checks passed
```

`graph build` already runs the audit and exits with status 1 if it fails (writing
`graph.rejected.json` instead of `graph.json` — see [Workspace layout](#workspace-layout)) — a
separate `ragent graph audit` call is only needed to re-check a graph later.

## Workspace layout

Everything `ragent` writes lives under the configured `workspace` (default `.ragent/`, override with
`--workspace` or `RAGENT_WORKSPACE`):

```text
.ragent/
  skills/<name>/SKILL.md                   # upstream-generated skill entry point
  skills/<name>/chapters/ch<NN>-<slug>.md   # upstream chapter artifacts
  skills/<name>/glossary.md                # supporting summaries, not graph inputs
  skills/<name>/patterns.md
  skills/<name>/cheatsheet.md
  graph.json                               # current compiled+audited stage graph
  graph.mmd                                # Mermaid rendering of graph.json
  graph.rejected.json                      # last build that failed audit (only written on failure)
  usage.json                               # lifetime + by-model token/cost ledger (budget.persist)
  graph_versions/<timestamp>/graph.json    # pre-merge graph snapshot, one per `refine merge` call
  runs/<run_id>/run.json                   # query, status, contract, models, budget, output paths
  runs/<run_id>/graph.json                 # exact effective graph used by the run
  runs/<run_id>/skill.md                   # exact explicit skill, when configured
  runs/<run_id>/trace.jsonl                # append-only transitions, usage, search, completion/errors
  runs/<run_id>/sources.json               # canonical fetched-source provenance and redirect aliases
  runs/<run_id>/sources/*.md               # immutable fetched evidence snapshots
  runs/<run_id>/artifacts/*.md             # per-stage artifacts
  runs/<run_id>/report.md                   # validated final report
  runs/<run_id>/verification.json           # latest offline verification result
  refine/<timestamp>/proposal.json          # `refine propose` output
  refine/<timestamp>/rollback.json         # `refine rollback` (gated acceptance) output
```

## Configuration

`load_config` builds the effective config in this exact precedence, each step overlaying the last:

1. Built-in defaults (`DEFAULT_CONFIG` in `src/ragent/config.py`).
2. `ragent.toml` in the current directory, or the path given by `--config`.
3. Environment variable overlay (see table below).
4. `--model role=provider/model` overrides (repeatable).
5. `--workspace PATH`, applied last, unconditionally.

Global CLI flags: `--config PATH`, `--workspace PATH`, repeatable `--model role=provider/model`.

Six roles are required — `graph_builder`, `executor`, `report`, `obsidian`,
`refiner`, and `judge`. Each role has `provider`, `model`, `temperature`, `max_tokens`, and optional
`reasoning_max_tokens`; an explicit reasoning cap must be positive and smaller than `max_tokens`.
Missing roles and unknown providers are configuration errors.

**Built-in defaults** (used for any role/field not set in `ragent.toml`): every role routes to
`openrouter` / `anthropic/claude-sonnet-4.5`, `temperature = 0.2`, `max_tokens = 8192`.

**This repo's checked-in `ragent.toml`**: every role routes to `openrouter` /
`z-ai/glm-5.3-flash`. `graph_builder` additionally sets `max_tokens = 16384` and
`temperature = 0.1`. Other roles inherit the default `temperature`/`max_tokens`.

PDF conversion is configured separately, not as an LLM role:

```toml
[book]
agent = ["hermes", "--skills", "book-to-skill", "-z", "{prompt}"]
```

`{prompt}` is replaced with the `/book-to-skill` request and pre-answered conversion
questions. The upstream skill must already be installed in that host. Other host
agents require changing this command; ragent passes its interpreter as `PYTHON_BIN`,
prepends its directory to `PATH`, and disables optional extractor-package installs.

**Illustrative multi-provider example** — mix providers per role, e.g. to keep the judge on a
cheaper/different model than the executor:

```toml
workspace = ".ragent"

[providers.openrouter]
kind = "openrouter"
api_key_env = "OPENROUTER_API_KEY"

[providers.openai]
kind = "openai"
api_key_env = "OPENAI_API_KEY"

[roles.executor]
provider = "openrouter"
model = "anthropic/claude-sonnet-4.5"

[roles.judge]
provider = "openai"
model = "gpt-5-mini"
temperature = 0.0
```

Other config sections:

```toml
[research]
query = "Optional reusable query"
graph_path = ".ragent/graph.json"
skill_path = ".ragent/skills/paper/SKILL.md"
instructions = "Review-specific requirements"
min_sources = 6
max_steps = 24
report = true
obsidian = true

[search]
backend = "ddg"                 # or "tavily"
api_key_env = "TAVILY_API_KEY"
max_results = 6

[obsidian]
backend = "vault"               # bundle requires vault, not MCP
vault_path = "/path/to/existing/vault"
folder = "Research"
layout = "bundle"                # "note" or deterministic "bundle"
tags = ["research"]
related_notes = ["Existing/Note"]
mcp_command = []                 # single-note MCP only

[budget]
strict = true
cost_limit_usd = 3.50
warn_fraction = 0.8
persist = true                  # required by strict mode

[prices."openai/gpt-5-mini"]    # ordinary mode only when provider cost is absent
input_per_mtok = 0.25
output_per_mtok = 2.00
```

Strict mode requires every effective role and node override to use OpenRouter. Bundle export requires
`research.report = true`, `obsidian.backend = "vault"`, and an existing vault containing
`.obsidian`. `ragent.e2e.toml` is the checked-in, non-secret profile for the process-aware research
scenario; it never contains an API key.

Environment overlay variables: `RAGENT_WORKSPACE`, `RAGENT_SEARCH_BACKEND`,
`RAGENT_OBSIDIAN_VAULT_PATH`, `RAGENT_OBSIDIAN_FOLDER`, `RAGENT_MODEL_<ROLE>` (must be
`provider/model`, role name upper-cased), `RAGENT_BUDGET_TOKENS`, `RAGENT_BUDGET_COST_USD`.

One-off role override from the CLI:

```bash
ragent --model executor=openai/gpt-5-mini research "query"
```

## CLI command reference

| Command | Flags (defaults) | Notes |
|---|---|---|
| `ragent init` | — | Writes `ragent.toml` (if absent), `.env.example`, and `skills/`, `runs/`, `graph_versions/`, `refine/` under the workspace. |
| `ragent providers check` | — | Pings every configured provider's `/models` endpoint; prints provider status and a role → provider/model table; exits 1 if any provider failed. |
| `ragent book PDF` | `--out PATH` (skills root), `--name SLUG` (source stem), `--mode text\|technical` (`text`), `--depth study\|reference` (`study`), `--force` | Runs upstream conversion via `[book].agent`, writing `<out>/<name>/`. Complete existing output is cached; incomplete output fails. `--force` deletes and regenerates that named skill. |
| `ragent graph build` | `--skill-dir PATH`, `--out PATH`, `--no-extend` | Compiles upstream `chapters/ch<NN>-*.md` in numeric order. Defaults to the sole generated skill under `skills/`; multiple skills require `--skill-dir`. Supporting files are validated but not compiled. `--no-extend` needs no skill or LLM. Writes `graph.json` + `graph.mmd`, or `graph.rejected.json` on audit failure. |
| `ragent graph show` | `--node TEXT`, `--graph PATH` | Prints a tree of one node (or all nodes) with its outward edges and their metrics. |
| `ragent graph audit` | `--graph PATH` | Re-runs the soundness audit on a saved graph; exits 1 on failure. |
| `ragent research [QUERY]` | `--graph PATH`, `--start TEXT`, `--max-steps N`, `--report/--no-report`, `--obsidian/--no-obsidian`, `--preflight` | Resolves omitted values from `[research]`. `--preflight` performs graph/route/skill/credential/model/price/budget/vault checks without creating a run, invoking inference, or writing notes. |
| `ragent runs list` | — | Table of run id, transition count, and explicit persisted state for new runs. |
| `ragent runs show RUN_ID` | — | Prints saved `run.json`, the raw trace, and output/verification paths. Run IDs are containment-checked. |
| `ragent runs verify RUN_ID` | — | Offline verification of recorded provenance, stages, fetched source hashes, report citations, bundle notes/links/hashes, model identity, strict exposure, and completion evidence. Exit 0 means every recorded requirement passed. |
| `ragent runs export CSV_PATH` | — | Writes transition-only `case_id,activity,timestamp,step,edge_id,metric_passed` rows. |
| `ragent refine propose` | — | Sends aggregated trajectory stats (not the graph, artifacts, or query text) to the `refiner` role; writes `refine/<ts>/proposal.json`. |
| `ragent refine merge PROPOSAL` | `--accept ID` (repeatable), `--all` | Applies selected proposal operations, snapshotting the current graph first; rejects if the merged graph fails audit. |
| `ragent refine rollback CANDIDATE` | — | Runs the gated-acceptance comparison (baseline vs. candidate graph over three fixed eval queries) and rolls back the candidate if it doesn't clear the acceptance thresholds. |
| `ragent tui` | — | Launches the interactive three-pane UI. |

Exit codes: expected failures (`RagentError` subclasses, including `GraphError`, `MetricError`,
`ProviderError`, `BudgetError`, `PreflightError`, `ExportError`, and `VerificationError`) print
`error: <message>` and return 1 from `entrypoint()`.
`providers check` and a failing graph build/audit exit 1 via a separate `typer.Exit(1)`. Malformed
CLI arguments or an invalid `--model` value are Typer usage errors, exit code 2.

See [algorithm.md](docs/algorithm.md) for exactly how `research` walks the graph and
[trajectories.md](docs/trajectories.md) for what `runs`/`refine` read and write.

## TUI reference

`ragent tui` launches a Textual app with:

- **Budget strip** (`#budget`) — the live `UsageLedger.status_line()`.
- **Query input** (`#query`) — placeholder `Research query (or PDF path for Book)`; also where slash
  commands are typed.
- **Command palette** (`#palette`) — hidden until the query input starts with `/`, then filtered as
  the command is typed.
- **Role table** (`#models`: Role / Provider / Model) next to the **model override input**
  (`#model_override`, placeholder `role=provider/model, then Enter`) — changes are in-memory only.
- **Graph tree** (`#graph`, 34% width) — nodes as `id — title`, edges as
  `edge_id → target  [metric_kind:metric_key]`; loaded from `.ragent/graph.json` if present, else the
  seed graph. Selecting any node or edge row renders its full contents (prompt template, bound
  tools, produced artifact key, gate metric, and `on_fail`) in the artifact pane.
- **Event log** (`#log`).
- **Artifact pane** (`#artifact`) — the newest artifact or selected graph row, rendered as Markdown.
- **Per-model breakdown table** (`#breakdown`: Model / Calls / In / Out / Total / Cost) — hidden
  until toggled with `u` or `/budget`.

### TUI screenshots

Historical screenshots below predate the six-role host-agent cutover; the current
role table excludes `book_to_skill`. Selecting an edge exposes its gate, retry
behavior, context, tools, and prompt:

![TUI graph with the frame_goal edge selected and its details visible](docs/assets/tui-edge-detail.svg)

Typing `/` opens the filtered command palette:

![TUI slash-command palette listing all available commands](docs/assets/tui-command-palette.svg)

Key bindings:

| Key | Action | Behavior |
|---|---|---|
| `r` | Run | Reads `#query` as the research query. A leading `/` logs `slash commands run on Enter, not r` instead of running. Empty input logs `enter a research query`. |
| `b` | Book | Treats `#query` as a PDF path, runs upstream conversion with streamed host output, and displays the generated `SKILL.md`. A missing/invalid path logs `Book binding uses the query field as a PDF path`. |
| `g` | Build graph | Rebuilds `graph.json` from the sole generated skill under `skills/` with no progress streaming. Unlike the CLI, this writes no `graph.mmd` and prints no audit. |
| `p` | Book→Graph | Runs the full book→graph pipeline, but only if `#query` already holds a PDF path; otherwise logs `pipeline: type /book <pdf-path> and press Enter`. |
| `a` | Audit | Audits the currently loaded (or seed) graph and logs every finding. |
| `u` | Budget | Toggles the `#breakdown` table and refreshes it. |
| `q` | Quit | — |

Slash commands (typed into `#query`, submitted with Enter):

Typing `/` opens the palette, filtered as you type. `up`/`down` move the
highlight, `tab` completes the highlighted command into the query box,
`escape` dismisses it, clicking an entry completes it, and `enter` still
submits.

- `/book <pdf-path>` — runs upstream conversion with streamed host output, then compiles
  its `chapters/ch<NN>-*.md` files with live per-chapter log lines. The graph
  tree resets to the seed and grows live as each `graph_delta` arrives. On completion it writes
  `graph.json` (or `graph.rejected.json` if the audit fails), always writes `graph.mmd`, and shows
  the Mermaid source in the artifact pane. The path is split on whitespace (`raw.split()`), so paths
  containing spaces are not supported.
- `/budget` — prints actual session/campaign spend, outstanding reserved exposure, remaining strict
  allowance, and per-model totals.
- `/budget tokens N` — sets the ordinary session token limit.
- `/budget cost USD` — changes an ordinary cost limit. In strict mode only a lower limit above current
  exposure is accepted; increases are refused.
- `/budget price provider/model IN OUT` — sets a manual ordinary-mode per-mtok price.
- `/budget reset` — resets display/session totals only; persisted strict campaign exposure is unchanged.
- `/budget off` — clears ordinary limits and is refused in strict mode.
- Any other `/…` prints the two help lines: the budget usage summary and `/book <pdf-path>`.

`/budget` and `#model_override` edits never write `ragent.toml` — they only mutate the in-memory
`Config`/`UsageLedger` for the current TUI session.

The command palette is the discoverable path for slash commands. The `p`
binding still only works when a PDF path is already sitting in `#query`;
otherwise use `/book <pdf>` and Enter.

## Budgets and cost control

Ordinary mode retains session `precheck()` then post-response `record()` behavior. Strict mode uses a
version-2 persistent campaign ledger instead:

1. authenticated preflight validates exact OpenRouter model pricing/capabilities and eligible
   endpoints;
2. each `LLM.complete` atomically reserves the conservative maximum cost under a sidecar `flock`
   before its single HTTP POST;
3. a successful response must report finite nonnegative cost and meaningful token usage within the
   validated bounds; settlement replaces the reservation with actual usage in one atomic write;
4. any timeout, malformed response, missing usage, interruption, or accounting failure leaves the
   reservation outstanding and blocks further paid calls.

Admission requires `lifetime actual + outstanding reservations + new reservation <=
strict_cost_limit_usd`. The campaign ceiling is persisted and cannot be raised or bypassed by a new
process, `/budget reset`, `/budget off`, a malformed ledger, or a failed write. Strict OpenRouter
requests disable transport/schema retries and constrain provider routing, parameters, and maximum
prices to the preflight snapshot.

`ragent.e2e.toml` sets a `$3.50` strict ceiling for the process-aware scenario. Run the free checks
before inference:

```bash
ragent --config ragent.e2e.toml graph audit --graph .ragent/graph.json
ragent --config ragent.e2e.toml research --preflight
```

This proves admission and configuration only. The completed run plus `runs verify` proves the
specific end-to-end scenario exercised; it is not a claim that every optional graph branch or every
application feature was tested.

## Further reading

- [docs/algorithm.md](docs/algorithm.md) — the executor's run loop, tool loop, metric kinds, and budget enforcement.
- [docs/skill-graph.md](docs/skill-graph.md) — what a skill graph is, the shipped seed, its schema, and its soundness rules.
- [docs/trajectories.md](docs/trajectories.md) — trajectory records, export/aggregation, and offline graph refinement.
- [docs/book-to-skill.md](docs/book-to-skill.md) — PDF extraction, chapter segmentation, distillation, and the chapter→graph merge.
- [docs/README.md](docs/README.md) — index of the above.
