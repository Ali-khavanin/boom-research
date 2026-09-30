# The Skill Graph

## 2.1 What it is here

A skill graph is a directed graph whose nodes are research stages and whose
edges are metric-gated transitions. Concretely, in `src/ragent/graph_builder/schema.py`
an `Edge` carries, on the same object, everything needed to execute that
transition:

- `prompt_template` — the instruction sent to the executor LLM for this step.
- `tool_set` — the allowlist of tool names bound for this step (empty by
  default; the executor cannot call anything outside it).
- `produces` — the artifact key the step's output is stored under.
- `termination_metric` — a `Metric` that is evaluated against the produced
  artifact; the edge is only taken once this passes.

A `Node` is just an id, title, description, an optional `terminal` flag, an
optional per-node `model` override, and optional `provenance`. All of the
procedural content — what to do, what tools are allowed, what "done" means for
this step — lives on the edge, not the node.

This is a deliberate contrast with free-form chain-of-thought prompting. In a
chain-of-thought agent the reasoning structure is implicit in a single prompt
and is improvised fresh on every call; nothing stops the model from skipping a
stage, inventing an unapproved shortcut, or asserting success without
evidence. Here the reasoning structure is data: it is `graph.json`, a
validated document that exists before any run starts. It can be inspected and
statically checked with `ragent graph audit` (see §2.4) independently of
executing it, and at run time the executor (see [algorithm.md](algorithm.md))
does not ask the model whether a stage succeeded — it deterministically
selects the first eligible outward edge in the edges' declared order and
gates every transition on `termination_metric` matching the artifact that was
actually written. The graph is compiled once, then enforced, not
re-improvised per call.

### What an edge holds

An edge is the unit that carries both halves of a transition: the context
handed to the model, and the verification that must pass before control moves
to `target`.

| group | field | role |
|---|---|---|
| **identity** | `id` | Stable identifier for the transition. |
| **identity** | `source` | Node from which the edge is eligible to run. |
| **identity** | `target` | Node that receives control after verification passes. |
| **context in** | `prompt_template` | Instruction rendered for the executor model, including the run context placeholders it uses. |
| **context in** | `tool_set` | Exact allowlist of tools bound for this transition; an empty list means prompt only. |
| **context in** | `precondition` | Optional metric that must already pass for this edge to be eligible. |
| **output contract** | `produces` | Artifact key under which the model's output is stored. |
| **verification / control flow** | `termination_metric` | Metric that the produced artifact must pass before control advances. |
| **verification / control flow** | `on_fail` | Optional node to route back to after the allowed failed attempts. |
| **verification / control flow** | `max_attempts` | Number of failed attempts allowed before failure routing or hard-error escalation. |
| **metadata** | `provenance` | Optional source metadata, such as the chapter and cue from which the edge was extracted. |

For example, the shipped seed contains this `review_prior_work` edge verbatim:

```json
    {
      "id": "review_prior_work", "source": "goal", "target": "what_has_been_done",
      "prompt_template": "Research prior work relevant to {query} and the goal. Use search and fetch, cite at least three distinct source URLs directly in the synthesis, and save a note when useful. Artifacts: {artifacts}. Last failure: {last_failure}",
      "tool_set": ["browser.search", "browser.fetch", "obsidian.note"],
      "termination_metric": {"kind": "has_citations", "key": "prior_work", "n": 3},
      "produces": "prior_work", "on_fail": "goal"
    },
```

The handed-over context is the prompt template with `{query}`, `{artifacts}`,
and `{last_failure}`, plus exactly three tools: `browser.search`,
`browser.fetch`, and `obsidian.note`. Verification requires at least three
distinct URLs inside the `prior_work` artifact. If that verification still
fails, control returns to `goal`.

Selecting an edge row in `ragent tui`'s graph tree renders exactly these
fields in the artifact pane.

## 2.2 The shipped seed

`src/ragent/data/seed_graph.json`, loaded by `load_seed()` in
`src/ragent/graph_builder/seed.py`, defines eight nodes and seven edges with
`version: 1` and `entry: "start"`.

Nodes, in file order, with their `description` fields quoted verbatim:

| id | description |
|---|---|
| `start` | `Receive and frame the research query.` |
| `goal` | `State the research objective and scope.` |
| `what_has_been_done` | `Find and synthesize prior work.` |
| `limitations` | `Identify limitations in prior work.` |
| `gaps` | `Derive actionable research gaps.` |
| `feasibility` | `Assess data and method availability.` |
| `quick_test` | `Design a fast discriminating test.` |
| `done` | `Research report and notes are complete.` |

`done` is the only node with `"terminal": true`.

Edges, in file order:

| id | source → target | tool_set | termination_metric | produces | on_fail |
|---|---|---|---|---|---|
| `frame_goal` | `start` → `goal` | *(default)* | `min_words`, key `goal`, `n: 60` | `goal` | *(none)* |
| `review_prior_work` | `goal` → `what_has_been_done` | `browser.search`, `browser.fetch`, `obsidian.note` | `has_citations`, key `prior_work`, `n: 3` | `prior_work` | `goal` |
| `identify_limitations` | `what_has_been_done` → `limitations` | `browser.fetch` | `min_items`, key `limitations`, `n: 3` | `limitations` | `what_has_been_done` |
| `derive_gaps` | `limitations` → `gaps` | *(default)* | `min_items`, key `gaps`, `n: 3` | `gaps` | `limitations` |
| `assess_feasibility` | `gaps` → `feasibility` | `browser.search` | `llm_rubric`, key `feasibility`, rubric `each gap has data/method availability verdict` | `feasibility` | `gaps` |
| `design_quick_test` | `feasibility` → `quick_test` | *(default)* | `min_words`, key `quick_test`, `n: 120` | `quick_test` | `feasibility` |
| `publish_results` | `quick_test` → `done` | `report.generate`, `obsidian.note` | `artifact_exists`, key `report_path` | `report_path` | `quick_test` |

*(default)* means the field is absent from the JSON for that edge and takes
its schema default. Every one of the seven seed edges also omits
`precondition`, `max_attempts`, and `provenance`, and every one of the eight
seed nodes omits `model` and `provenance`. Per `schema.py` those omissions
resolve to: `tool_set = []`, `precondition = None`, `max_attempts = 2`, no
`provenance` (`None`), and no per-node `model` override (`None`) — i.e. every
seed edge retries up to twice with no per-node LLM override anywhere in the
seed.

## 2.3 Schema reference

All model definitions live in `src/ragent/graph_builder/schema.py`. Field
lists below are exactly the declared pydantic fields, in declaration order.

**`RoleOverride`** (no `extra` restriction declared):
| field | type | default |
|---|---|---|
| `provider` | `str \| None` | `None` |
| `model` | `str \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `max_tokens` | `int \| None` | `None` |

**`Metric`** (no `extra` restriction declared):
| field | type | default |
|---|---|---|
| `kind` | `Literal["artifact_exists", "min_words", "min_items", "has_citations", "regex", "llm_rubric"]` | required |
| `key` | `str` | `"output"` |
| `n` | `int \| None` | `None` |
| `pattern` | `str \| None` | `None` |
| `rubric` | `str \| None` | `None` |

A `model_validator(mode="after")` named `validate_parameters` enforces
conditional requirements after the base fields parse:
- `kind in {"min_words", "min_items", "has_citations"}` and `n is None` →
  `ValueError(f"metric {self.kind} requires n")`.
- `kind == "regex"` and `not self.pattern` →
  `ValueError("regex metric requires pattern")`.
- `kind == "llm_rubric"` and `not self.rubric` →
  `ValueError("llm_rubric metric requires rubric")`.

`artifact_exists` has no conditional requirement — it only ever needs `key`.

**`Edge`** — `model_config = ConfigDict(extra="forbid")`:
| field | type | default |
|---|---|---|
| `id` | `str` | required |
| `source` | `str` | required |
| `target` | `str` | required |
| `prompt_template` | `str` | required |
| `tool_set` | `list[str]` | `Field(default_factory=list)` |
| `precondition` | `Metric \| None` | `None` |
| `termination_metric` | `Metric` | required |
| `produces` | `str` | `"output"` |
| `on_fail` | `str \| None` | `None` |
| `max_attempts` | `int` | `Field(default=2, ge=1)` |
| `provenance` | `dict[str, Any] \| None` | `None` |

**`Node`** — `model_config = ConfigDict(extra="forbid")`:
| field | type | default |
|---|---|---|
| `id` | `str` | required |
| `title` | `str` | required |
| `description` | `str` | `""` |
| `terminal` | `bool` | `False` |
| `model` | `RoleOverride \| None` | `None` |
| `provenance` | `dict[str, Any] \| None` | `None` |

**`Graph`** — `model_config = ConfigDict(extra="forbid")`:
| field | type | default |
|---|---|---|
| `version` | `int` | `1` |
| `entry` | `str` | `"start"` |
| `nodes` | `list[Node]` | required |
| `edges` | `list[Edge]` | required |

`Graph` also exposes `out_edges(node_id) -> list[Edge]` (all edges with
matching `source`, in `graph.edges` declaration order) and
`node(node_id) -> Node` (raises `GraphError(f"graph node does not exist: {node_id}")`
if no node matches). `Graph`, `Node`, and `Edge` all set `extra="forbid"`, so
any stray key in a hand-edited or model-produced graph document fails
validation instead of being silently dropped; `Metric` and `RoleOverride` do
not set `extra="forbid"`.

`slug_id(value: str) -> str` normalizes an arbitrary string into a node/edge
id: `re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")`, falling back to
the literal string `"stage"` if that produces an empty string.

## 2.4 Soundness rules

`audit(graph: Graph) -> Audit` in `src/ragent/graph_builder/validate.py` runs
a fixed sequence of structural checks and returns every violation as a
`Finding(level, code, detail)`. In source order:

| code | trigger | detail template |
|---|---|---|
| `duplicate_node` | `len({node.id for node in graph.nodes}) != len(graph.nodes)` | `node ids are not unique` |
| `duplicate_edge` | `len({edge.id for edge in graph.edges}) != len(graph.edges)` | `edge ids are not unique` |
| `missing_entry` | `graph.entry` is not among the node ids | `entry id is unresolved: {graph.entry}` |
| `terminal_count` | number of nodes with `terminal=True` is not exactly `1` | `expected exactly one terminal node, found {n}` |
| `bad_source` | an edge's `source` is not a known node id | `edge {edge.id} source is unresolved: {edge.source}` |
| `bad_target` | an edge's `target` is not a known node id | `edge {edge.id} target is unresolved: {edge.target}` |
| `bad_on_fail` | an edge has a non-empty `on_fail` that is not a known node id | `edge {edge.id} on_fail is unresolved: {edge.on_fail}` |
| `dead_end` | a non-terminal node has zero outward edges (`graph.out_edges(node.id)` empty) | `non-terminal node has no outward edge: {node_id}` |
| `unreachable` | node id not reached by forward BFS from `graph.entry` | `node is unreachable from entry: {node_id}` |
| `cannot_reach_done` | node id not reached by reverse BFS rooted at the literal id `"done"` | `done is not reachable from node: {node_id}` |
| `terminal_unreachable` | the (first) terminal node id is not in the forward-reachable set from entry | `terminal node is unreachable: {terminals[0]}` |
| `sound` (info) | emitted only when the `findings` list is otherwise empty | `graph reachability and dead-end checks passed` |

Two details worth calling out explicitly because they affect what a passing
audit actually guarantees:

- **`unreachable` follows only primary `target` links, never `on_fail`.** The
  forward BFS enqueues `edge.target for edge in graph.out_edges(node_id)`;
  `edge.on_fail` is never read during this walk. A node reachable *only*
  through an `on_fail` back-edge from elsewhere is still flagged
  `unreachable` if no primary `target` chain from `entry` reaches it.
- **`cannot_reach_done` is rooted at the hardcoded id `"done"`, not at
  whichever node happens to have `terminal=True`.** The reverse BFS starts
  with `deque(["done"])` guarded by `if "done" in ids`. If a graph's terminal
  node were renamed away from the id `"done"`, this check would silently stop
  finding anything reachable (every non-`"done"`-id node would report
  `cannot_reach_done`), which is a different failure mode from
  `terminal_unreachable` (that one does track whichever node is
  `terminal=True`, via the *forward* reachable set from `entry`).

The returned `Audit` model carries: `entry` (`graph.entry`), `node_count`
(`len(graph.nodes)`), `done_reachable_count` (size of the reverse-from-`"done"`
reachable set), `dead_ends` (sorted list of node ids with no outward edge),
`unreachable` (sorted list of node ids missed by the forward BFS), and
`findings` (the full `Finding` list in the order above). `Audit.ok` is a
property: `not any(finding.level == "error" for finding in self.findings)` —
true iff no error-level finding was produced (the `sound` finding is
`level="info"` and never affects `ok`).

This matters because the executor (see [algorithm.md](algorithm.md)) performs
no fallback when it lands on a node with no eligible outward edge or when it
exhausts the graph without reaching a terminal node — it raises `GraphError`
or `MetricError` and the run stops mid-way, artifacts already spent. A `dead_end`
or `unreachable` node found by `audit()` before execution is exactly the
condition that would strand a live run; running `ragent graph audit` (or the
audit `ragent graph build` runs automatically) catches this before any tokens
are spent on a graph that cannot finish.

## 2.5 Extending the graph

Upstream `book-to-skill` chapters (`chapters/ch<NN>-*.md`; see
[book-to-skill.md](book-to-skill.md)) are compiled in numeric chapter order and
merged into the seed graph rather than building a separate graph per chapter.
The skill entry point and supporting summaries are validated but not compiled.
Three properties of that merge keep the result auditable:

- **Seed id reuse.** Node and edge ids produced from chapter text are passed
  through `slug_id`, and a proposed node is only added if its normalized id
  has not already been seen (starting from the seed's ids). This means a
  chapter that restates "identify limitations in prior work" in its own words
  reuses the seed's `limitations` node instead of forking a parallel copy —
  chapters overlap onto the same stage graph rather than each contributing an
  independent one.
- **Provenance tagging.** Every node or edge that is *not* part of the seed
  carries `provenance = {"chapter": ..., "cue": ...}`, recording which
  chapter file (for example `ch01-introduction.md`) and which exact phrase
  produced it. Seed nodes/edges have no `provenance` (see §2.2).
- **Deterministic pruning.** After each chapter's additions are merged, the
  graph is re-audited, and any newly added node this chapter is responsible
  for that the audit flags as `dead_end`, `unreachable`, or
  `cannot_reach_done` is removed again, along with edges incident to it — so
  a chapter can only ever leave the graph as sound as it found it.

The full merge algorithm — sanitization of model output, id collision rules,
the fixed-point pruning loop, and the emitted progress events — is documented
in [book-to-skill.md](book-to-skill.md).

## 2.6 Related work

This subsection is external context, not the basis of this implementation —
none of this repo's code was derived from the papers below. Three arXiv
identifiers were checked by fetching their abstract pages directly; all three
resolve.

- **Graph-of-Skills** ([2604.05333](https://arxiv.org/abs/2604.05333),
  Liu et al., 2026) builds an executable skill graph offline from a *library*
  of skill packages, then at inference time retrieves a bounded,
  dependency-aware bundle of skills for a task via hybrid semantic-lexical
  seeding and reverse-aware Personalized PageRank over that graph. The graph
  here is a retrieval index over many pre-existing skills, not something
  compiled from one source document.
- **SkillGraph** ([2605.12039](https://arxiv.org/abs/2605.12039), Li et al.,
  2026) represents reusable skills as nodes in a directed graph with typed
  edges encoding prerequisite, enhancement, *and* co-occurrence relations
  (three relation types, not just prerequisite/enhancement), continuously
  updated from agent trajectories and reinforcement-learning feedback so the
  library and the agent's policy improve together.
- **AIP** ([2606.04781](https://arxiv.org/abs/2606.04781), Blumenfeld &
  Webber, 2026) models a single skill itself as a directed execution graph —
  discrete steps as nodes backed by deterministic scripts or natural-language
  descriptions, connected by typed input/output edges, governed by a
  schema-validated specification — so that failures can be diagnosed and
  repaired node-by-node instead of by rewriting prose.

The differentiator from all three: this repo does not retrieve or learn over
a pre-existing library of skill graphs. It compiles exactly one graph from a
single input methodology document (§2.5), and every transition in that graph
is a deterministic metric gate (§1.4 in [algorithm.md](algorithm.md)) with its
own per-edge tool allowlist, checked against a materialized artifact rather
than against an agent's self-report of success.
