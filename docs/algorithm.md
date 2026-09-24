# Algorithm

This page documents the executor: the code that walks an audited [skill graph](skill-graph.md) node
by node, calling an LLM with a bound tool set at each edge and gating advancement on a metric
check against the artifact the LLM produced. Every claim below is verified against
`src/ragent/executor/runner.py`, `src/ragent/executor/metrics.py`, `src/ragent/executor/context.py`,
`src/ragent/tools/registry.py`, `src/ragent/tools/browser.py`, `src/ragent/tools/report_generator.py`,
`src/ragent/tools/obsidian_mcp.py`, `src/ragent/llm/usage.py`, and `src/ragent/data/seed_graph.json`.

## 1.1 The run loop

`run(graph, query, cfg, *, start=None, max_steps=None, report=None, obsidian=None,
on_event=None) -> RunResult` resolves omitted execution switches from `cfg.research`.
`prepare_graph` makes an effective copy: it removes graph-driven Obsidian publication for bundle
layout, removes disabled report/Obsidian tools, and raises the `prior_work` citation gate to at least
`research.min_sources`.

Before a run directory or paid call exists, `preflight(prepared_graph, effective_cfg)` audits graph
structure, inspects the first-edge route without calling rubric metrics, validates route prompts/tool
bindings, reads an explicit skill, and checks only the providers/outputs the run needs. Strict mode
also authenticates the OpenRouter key, validates exact models/prices/capabilities/endpoints, and
checks the persistent campaign ledger. A startup failure therefore creates no fake failed run.

After admission, the executor creates `<workspace>/runs/<run_id>/`, copies the effective graph and
skill, and writes version-1 `run.json` with status `running`. Edge choice remains declaration-order
deterministic: every precondition is checked and the first eligible edge runs. Node model overrides
include `reasoning_max_tokens` and are validated after merging with the executor role.

Text artifacts are written under `artifacts/`. A key ending in `_path` never falls back to model
text: it is populated only by a real, nonempty file beneath the run directory or configured vault.
`BudgetError` escapes the tool loop immediately; ordinary retrieval errors are returned to the model
for bounded recovery.

For an edge targeting the terminal node, requested outputs are completed before its metric can pass:
the report is generated/validated, a requested bundle is exported/validated, `run.json` output paths
and accounting are updated, and `verify_outputs` must pass. Only then is the terminal transition
logged. A successful run appends `event=\"complete\"`, persists status `completed`, and runs
`verify_run`; any later verification failure changes status to `failed` and records an error.

The retry and fallback rules remain edge-local: failures increment `ctx.attempts[edge.id]`, populate
`last_failure`, jump to `on_fail` at the configured threshold, and otherwise retry until the existing
hard-stop rule fires. `RunResult` includes artifacts, citations, steps, and the final verification
result; `reached_done` is true only after the output path above succeeds.

## 1.2 Prompt rendering

`_render` supplies the existing `{query}`, `{node}`, `{artifacts}`, and `{last_failure}` replacements
plus `{skill}`, `{instructions}`, and `{sources}`. The full configured skill procedure, research
instructions, and canonical source index are also appended even when an older prompt template lacks
the new placeholders. The graph and edge order are not regenerated.

## 1.3 Tool loop and evidence

The loop allows at most six model/tool rounds, exposes only the edge's bound tools, and converts
dotted tool names to `__` wire names. Unknown or ordinary tool errors become structured tool
evidence; `BudgetError` is re-raised. After six rounds one final no-tool completion is requested.

`browser.search` records an `event=\"search\"` row with query, backend, count, and returned URLs.
Empty/challenge DDG responses are errors. `browser.fetch` accepts only HTTP(S), extracts HTML,
plain-text/Markdown, or readable PDFs, rejects empty evidence, and retains at most 12,000 characters.
Each successful canonical URL is snapshotted at
`sources/<sha256(canonical_url)[:16]>.md`; `sources.json` records original/final URLs, title,
retrieval timestamp, evidence path/hash, and redirect aliases. Citations are added only after both
files succeed. Redirect duplicates add an alias without replacing the original snapshot.

`report.generate` requires all six nonempty stages and at least `research.min_sources`
fetched-and-cited sources. Unknown URLs are rejected, aliases are canonicalized, and a valid report
is cached within the run to avoid another polishing call.

`obsidian.note` remains the graph-driven single-note/MCP API and raises export errors on
misconfiguration. `export_bundle(ctx)` is deterministic and uses no LLM: it stages
`index.md`, `report.md`, six stage notes, source excerpt notes, and `manifest.json` under the fresh
`<vault>/<folder>/<run_id>/` directory. Tags/frontmatter/links are code-generated; unallowlisted
model wikilinks become visible plain text. Every generated link and hash is validated before rename.
Related notes are hash-checked before/after and are never edited. Same-run re-export is a no-op only
when the manifest and every note hash still match.

## 1.4 Metric kinds

`check(metric, ctx, cfg=None)` retains `min_words`, `min_items`, `regex`, and `llm_rubric`.
`artifact_exists` treats ordinary keys as nonempty text, but `_path` keys pass only for an existing,
nonempty file inside the run or configured vault. `has_citations` normalizes scheme/host case,
drops fragments, preserves path/query semantics, resolves recorded redirects, and counts only URLs
whose fetched evidence record exists in `ctx.citations`. Invented URLs do not count.

The `prior_work` gate is raised during shared graph preparation to
`max(graph_threshold, research.min_sources)`, so insufficient evidence fails where search/fetch are
available rather than at publication.

`llm_rubric` still calls the configured judge with only the rubric and artifact. Its paid usage is
included in the transition delta because the runner takes the post-attempt accounting snapshot after
metric evaluation.

## 1.5 Budget enforcement

`UsageLedger` remains the only accounting system. Ordinary mode retains session `precheck` and
post-response `record`. Strict mode requires persistence, a positive finite dollar limit, OpenRouter
for every effective model, and Unix `fcntl` locking.

Strict admission is:

```text
lifetime actual cost + outstanding reservation bounds + new bound <= persisted campaign limit
```

`reserve(role, label, max_cost_usd)` reloads version-2 `usage.json` under a sidecar advisory lock,
rejects any existing reservation, and atomically persists the new bound before the HTTP POST.
`settle(reservation_id, usage)` requires meaningful nonnegative token usage and finite nonnegative
reported cost within both token and dollar bounds, adds actual usage, clears the reservation, and
atomically writes once. Settled IDs cannot settle twice.

Strict model preflight refreshes model and endpoint metadata. Eligible endpoints must support tools,
structured output, and reasoning; fit the output cap; charge no request/unsupported extra fee; and
stay at or below catalog prompt/completion prices. The reservation uses the maximum advertised
context multiplied by catalog prompt price plus effective `max_tokens` multiplied by completion
price. OpenRouter requests use `provider.only`, `max_price` in USD per million, `request=0`,
`require_parameters=true`, and `allow_fallbacks=false`.

A strict provider completion is exactly one HTTP POST: transport retries and the ordinary HTTP-400
schema-removal retry are disabled. Timeout, malformed response, missing usage, token/cost overflow,
interruption, or persistence failure leaves the maximum reservation outstanding. The next paid call
is blocked until the uncertainty is resolved externally; the application never guesses that an
unknown charge was free.

The campaign limit is persisted as a decimal string. A later process cannot raise it or open the
workspace non-strict; a lower limit is allowed only above current exposure. Version-1 totals migrate
only when they contain no unpriced calls. Malformed/incompatible/unwritable strict state fails closed.
Presentation snapshots convert decimals to numbers for existing trace/TUI consumers and expose
actual spend, reserved exposure, remaining allowance, and unresolved reservation count.

## 1.6 Worked trace

The shipped seed graph (`src/ragent/data/seed_graph.json`) defines the happy path the executor
walks for a fresh workspace. Quoting each edge's fields verbatim from the JSON:

| step | edge id | source → target | `tool_set` | `termination_metric` | `on_fail` |
|---|---|---|---|---|---|
| 1 | `frame_goal` | `start` → `goal` | *(none)* | `{"kind": "min_words", "key": "goal", "n": 60}` | *(none)* |
| 2 | `review_prior_work` | `goal` → `what_has_been_done` | `["browser.search", "browser.fetch", "obsidian.note"]` | `{"kind": "has_citations", "key": "prior_work", "n": 3}` | `goal` |
| 3 | `identify_limitations` | `what_has_been_done` → `limitations` | `["browser.fetch"]` | `{"kind": "min_items", "key": "limitations", "n": 3}` | `what_has_been_done` |
| 4 | `derive_gaps` | `limitations` → `gaps` | *(none)* | `{"kind": "min_items", "key": "gaps", "n": 3}` | `limitations` |
| 5 | `assess_feasibility` | `gaps` → `feasibility` | `["browser.search"]` | `{"kind": "llm_rubric", "key": "feasibility", "rubric": "each gap has data/method availability verdict"}` | `gaps` |
| 6 | `design_quick_test` | `feasibility` → `quick_test` | *(none)* | `{"kind": "min_words", "key": "quick_test", "n": 120}` | `feasibility` |
| 7 | `publish_results` | `quick_test` → `done` | `["report.generate", "obsidian.note"]` | `{"kind": "artifact_exists", "key": "report_path"}` | `quick_test` |

Walking it under §1.1's rules:

- **`start`** has one outward edge, `frame_goal`, with no `tool_set` (so no tools are bound — the
  model must write the goal purely from the rendered prompt) and no precondition, so it is always
  eligible. The artifact key is `goal`; passing requires at least 60 words. Since `frame_goal` has
  no `on_fail`, a run stuck here escalates straight to the hard `MetricError` on the 4th failure
  (`failures > max_attempts + 1 == 3`, using the schema default `max_attempts=2`).
- **`goal`** advances via `review_prior_work` into `what_has_been_done`, bound to
  `browser.search`, `browser.fetch`, and `obsidian.note`. The metric is `has_citations` with
  `n=3` against the `prior_work` artifact — at least three distinct URLs must appear in the
  synthesized text itself (per §1.4, calling `browser.fetch` alone does not satisfy this; the model
  must actually cite the URLs in the produced Markdown). Failing here loops back to `goal`
  (`on_fail: "goal"`), so a bad prior-work synthesis sends execution back to re-run `frame_goal`.
- **`what_has_been_done`** advances via `identify_limitations` into `limitations`, bound only to
  `browser.fetch` (no new search). Metric is `min_items` with `n=3` on the `limitations` key —
  at least three Markdown list items. `on_fail` loops back to `what_has_been_done`.
- **`limitations`** advances via `derive_gaps` into `gaps`, with no tools bound at all — the model
  must derive gaps purely from artifacts already in context (`{artifacts}` includes the prior
  `goal`, `prior_work`, and `limitations` text). Metric is `min_items` with `n=3` on `gaps`.
  `on_fail` loops back to `limitations`.
- **`gaps`** advances via `assess_feasibility` into `feasibility`, bound to `browser.search` only.
  Metric is `llm_rubric` with rubric text `"each gap has data/method availability verdict"` on key
  `feasibility` — judged by the `judge` role LLM per §1.4, not by a mechanical text check.
  `on_fail` loops back to `gaps`.
- **`feasibility`** advances via `design_quick_test` into `quick_test`, with no tools bound. Metric
  is `min_words` with `n=120` on key `quick_test`. `on_fail` loops back to `feasibility`. This is
  the edge spot-checked against `seed_graph.json`: its `termination_metric.kind` is `min_words`,
  `n` is `120`, and `on_fail` is the literal string `"feasibility"`.
- **`quick_test`** advances via `publish_results` into `done` (the graph's sole `terminal: true`
  node), bound to `report.generate` and `obsidian.note`. Metric is `artifact_exists` on key
  `report_path` — satisfied once `report.generate`'s `_path`-suffixed return value has been
  auto-promoted into `ctx.artifacts["report_path"]` by `_tool_loop` (§1.3). `on_fail` loops back to
  `quick_test`.

At every one of these seven hops, the model's job is only to produce text and optionally call the
listed tools; whether the run advances, retries in place, or falls back to an earlier node is
decided exclusively by `check()` re-reading the artifact the model just wrote, per the invariant
stated at the end of §1.1.
