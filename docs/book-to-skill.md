# Document to Skill Graph

Upstream `book-to-skill` owns the entire PDF-to-skill conversion: extraction,
chapter selection, and skill generation. `ragent` invokes a host agent following
upstream's `SKILL.md`, validates its artifacts, then compiles its chapters into
`graph.json`. See [algorithm.md](algorithm.md) for graph execution and
[skill-graph.md](skill-graph.md) for the graph schema and soundness rules.

## 4.1 Upstream ownership and installation

Both the extractor package and host skill are pinned to upstream commit
`c108d25b0cb58e1bdc361f3de02ed9f37075152f`. Release `v1.4.0` has no Hermes
support; this unreleased commit adds Hermes skill-root and extractor discovery.
Neither version exposes a Python/CLI skill-generation API: generation is the
host agent following [upstream's skill](https://github.com/virgiliojr94/book-to-skill/blob/c108d25b0cb58e1bdc361f3de02ed9f37075152f/SKILL.md).
There is no ragent segmentation or chapter-distillation fallback.

The Python dependency supplies extractor runtime packages (`pypdf`,
`pdfminer.six`, `pdf-inspector`) for ragent's interpreter. The host skill clone
is a separate prerequisite, outside the repository. Install Hermes, configure
its model credentials, and put `hermes` on `PATH`, then:

```bash
git clone https://github.com/virgiliojr94/book-to-skill.git ~/.hermes/skills/research/book-to-skill
git -C ~/.hermes/skills/research/book-to-skill checkout --detach c108d25b0cb58e1bdc361f3de02ed9f37075152f
hermes skills list
```

If the clone already exists, run
`git -C ~/.hermes/skills/research/book-to-skill fetch` instead of cloning,
then the same checkout. `hermes skills list` must show `book-to-skill`.
For an existing Python installation with the same upstream package version,
verify `direct_url.json`; pip may retain `v1.4.0` despite the changed revision:

```bash
.venv/bin/python -c "import importlib.metadata as m; print(m.distribution('book-to-skill').read_text('direct_url.json'))"
```

If it still names `v1.4.0`, replace just that package:

```bash
.venv/bin/pip install --force-reinstall --no-deps 'book-to-skill @ git+https://github.com/virgiliojr94/book-to-skill.git@c108d25b0cb58e1bdc361f3de02ed9f37075152f'
```

## 4.2 Host-agent invocation

`src/ragent/book_to_skill/agent.py` exposes
`generate_skill(source, skills_root, cfg, *, name=None, mode="text",
depth="study", force=False, on_event=None) -> SkillArtifacts`.
The source and skills root are expanded and resolved to absolute paths.
`mode` accepts `text` or `technical`; `depth` accepts `study` or `reference`.
The default name is the lowercased source stem with non-alphanumeric runs
replaced by `-`, stripped and truncated to 64 characters. Explicit names must
match `[a-z0-9]+(?:-[a-z0-9]+)*` and be at most 64 characters.

```toml
[book]
agent = ["hermes", "--skills", "book-to-skill", "-z", "{prompt}"]
```

The command must be nonempty and contain `{prompt}` in at least one argument.
Each occurrence is replaced with the full request; no shell is involved.
Hermes `-z` runs headlessly and `--skills` preloads the installed upstream skill.
Other hosts require changing this command and installing the same skill there.
Host-agent spend is separate from ragent's usage ledger and budget.

### Request (verbatim for the default `text` / `study` options)

`{source}`, `{slug}`, and `{skills_root}` are interpolated:

```text
/book-to-skill "{source}" {slug}

This is a non-interactive Full Conversion started by ragent; nobody can answer questions, so use these answers:
- Step 1.5 content type: 2 (Text-heavy), so BOOK_TYPE=text.
- Step 2.5 cost estimate: proceed with Full Conversion.
- Step 4 purpose: 4 (All of the above), so DEPTH=study.
- Step 5 destination: I explicitly request SKILLS_HOME="{skills_root}". Write the skill to "{skills_root}/{slug}/" exactly: no category subfolder, no symlink, no `hermes skills trust`, and nothing under ~/.hermes/skills or ~/.agents/skills.
- Missing optional extractor packages: do not install them; use the available fallback.
- Step 11 publish: skip.
```

`--mode technical` changes Step 1.5 to `1 (Technical), so BOOK_TYPE=technical`.
`--depth reference` changes Step 4 to
`3 (Reference specific chapters and concepts), so DEPTH=reference`.

The subprocess runs with `cwd=skills_root`, inherited environment plus
`PYTHON_BIN=sys.executable`, `BOOK_SKILL_INSTALL_MISSING=no`, and the interpreter's
directory prepended to `PATH`. Both interpreter settings make the upstream
extractor use ragent's virtual environment. stdout and stderr are combined,
decoded as UTF-8 with replacement, and streamed line by line. There is no
timeout; the user can interrupt. A nonzero exit raises
`book-to-skill agent exited with status N`. A missing executable raises an
actionable `PATH` error naming `[book].agent`.

| event | keys |
|---|---|
| `book_start` | `event`, `pdf`, `skill_dir`, `agent` |
| `agent_output` | `event`, `line` |
| `skill_done` | `event`, `path`, `chapters`, `cached` |

Subscriber exceptions are swallowed so a broken progress display does not
abort conversion. The CLI and TUI render agent output literally, not as Rich
markup. `ragent book` prints chapter/supporting-file counts, not a usage ledger.

## 4.3 Generated artifact contract

```text
<skills_root>/<slug>/
  SKILL.md
  chapters/ch<NN>-<slug>.md
  glossary.md
  patterns.md
  cheatsheet.md
```

Upstream's `SKILL.md` has `name` and `description` front matter. Chapters have
no front matter and follow Core Idea, Frameworks Introduced (When to use / How),
Key Concepts, Mental Models, Anti-patterns, optional Worked Example, Key
Takeaways, and Connects To.

`src/ragent/book_to_skill/skill.py` defines `SkillArtifacts(root, skill_file,
chapter_files, supporting_files)` and `load_skill(root)`. It requires `SKILL.md`
and all three supporting files to be files, plus at least one chapter matching
`^ch(\d+)-.+\.md$`. It ignores other chapter-directory entries and sorts by
`(int(chapter_number), filename)`, so `ch2` precedes `ch10`. Missing artifacts
raise `incomplete book-to-skill output in <root>: missing <names>`; an absent
chapter directory is reported as missing `chapters/ch<NN>-<slug>.md`.
This validates layout, not content quality or upstream heading completeness.

An existing target without `--force` is validated and returned with a cached
`skill_done` event, without invoking the host or checking source freshness.
An incomplete target is an error, not a cache miss. `--force` deletes that
target directory before generating the complete skill again.

`find_skill(skills_root)` discovers immediate `*/SKILL.md` candidates. Exactly
one is required; zero reports how to run `ragent book` or pass `--skill-dir`,
and multiple candidates are named in an error requesting `--skill-dir`.
The CLI resolves this default only when extending the graph.

Validate upstream content separately with:

```bash
.venv/bin/python ~/.hermes/skills/research/book-to-skill/tools/validate_skill.py --lens hermes .ragent/skills/08ijbas31/SKILL.md
```

## 4.4 Chapter to graph merge

`src/ragent/graph_builder/extract.py`'s `build_graph(skill_dir, llm, seed=None,
*, extend=True, on_event=None) -> Graph` folds each chapter's LLM-compiled
graph proposal into a running base graph, one chapter at a time, auditing and
pruning after each.

### Base graph and short-circuit

```python
base = (seed or load_seed()).model_copy(deep=True)
if not extend:
    return base
```

`extend=False` (`ragent graph build --no-extend`) returns the deep-copied seed
immediately — no chapter files are read, no LLM calls happen, no events fire,
and no audit runs.

### Per-chapter loop

Chapter files come from `load_skill(skill_dir).chapter_files`: a complete
upstream skill is required before any LLM call, and `ch<NN>-*.md` files are
processed in numeric chapter order. `SKILL.md`, `glossary.md`, `patterns.md`,
and `cheatsheet.md` are validated but not compiled: they index or summarize
the chapters and would duplicate their graph contributions.

### Per-chapter graph-compilation prompt (verbatim)

System message:

```
Compile one chapter file of a book-to-skill generated skill into a sound research-stage graph. The chapter follows the book-to-skill template: Core Idea, Frameworks Introduced (each with When to use and How), Key Concepts, Mental Models, Anti-patterns, optional Worked Example, Key Takeaways, and Connects To. Treat each framework's How steps as ordered stages, its When to use as the edge precondition or prompt intent, Anti-patterns as failure conditions that justify on_fail loop-backs, and Key Takeaways as termination criteria. Reuse the seed ids start, goal, what_has_been_done, limitations, gaps, feasibility, quick_test, done whenever applicable. Add a node only for a genuinely new stage. Every non-seed node and edge needs provenance with chapter and cue, where cue is an exact phrase copied from the chapter text. Metric field rules: kinds min_words, min_items, and has_citations each require an integer n (e.g. has_citations needs n = minimum citation count); kind regex requires pattern; kind llm_rubric requires rubric. Connectivity is mandatory: your JSON is merged as-is, with no cross-chapter wiring added afterward. Every node you include (seed or new) must sit on at least one edge path that starts at a seed node reachable from 'start' and ends at 'done'. Never add a node with no outgoing edge unless it is 'done' itself; never add a node with no incoming edge from 'start' or another node already on such a path. If a new node does not chain forward to 'done', omit it rather than leave it disconnected.
```

User message (`f"..."`, interpolating `sorted(node_ids)` — the ids merged so
far from the seed plus every earlier chapter this run — then the chapter
filename and its full source text):

```
Existing graph node ids already merged from other chapters (for id reuse only, do not redefine): {sorted(node_ids)}

Chapter: {chapter}
Return a complete Graph-shaped JSON proposal. It may repeat seed elements; deterministic merging will keep seed definitions.

{source}
```

The call goes through `llm.json(prompt, graph_schema(), validate_fn=lambda
value: Graph.model_validate(value), sanitize_fn=_sanitize_graph_proposal)`,
which is `ragent`'s `LLM.json` wrapper around `call_json` (see §4.5); the
result is re-validated as `Graph.model_validate(...)` once more on return.

### Deterministic merge rules

- **Node ids** normalize through `slug_id` (`_normalize_node` sets
  `node.id = slug_id(node.id)`, lowercasing and collapsing non-alphanumerics
  to `_`; see §4.5-adjacent `schema.py`).
- **Add-only-if-unseen**: `if node.id not in node_ids: base.nodes.append(node)`
  — a node whose normalized id already exists (seed or an earlier chapter's
  addition) is silently dropped, never overwritten. This is what makes seed
  and earlier-chapter node definitions always win over a later chapter's
  redeclaration.
- **Provenance overwrite**: `_normalize_node`/`_normalize_edge` unconditionally
  set `provenance = {"chapter": chapter, "cue": <preserved cue or "">}` on
  every proposed node/edge before merge — even if the model supplied its own
  `provenance.chapter`, it is replaced with the real source filename.
- **Edge id remapping**: proposal-local ids are mapped through
  `id_map = {node.id: slug_id(node.id) for node in proposal.nodes}` built
  from *this chapter's own proposal*, then `_normalize_edge` resolves
  `edge.source`/`edge.target`/`edge.on_fail` through that map (falling back to
  `slug_id(...)` directly if a referenced id was not itself a proposed node —
  e.g. a seed id).
- **Edge drop conditions** — an edge is silently dropped (never merged) if
  any of: its normalized `source` or `target` is not in the accumulated
  `node_ids` set (unknown endpoint); its normalized `id` already exists in
  `edge_ids`; or its `(source, target)` pair already exists in `pairs`
  (duplicate parallel edge between the same two nodes).

### Robustness layers

**`_sanitize_graph_proposal`** strips any key the model emitted outside the
schema's real shape before validation (some models ignore
`additionalProperties: false` and add stray fields, e.g. a `kind`
discriminator on a redeclared seed node). It filters against four fixed
allowlists:

```python
_GRAPH_KEYS = {"version", "entry", "nodes", "edges"}
_NODE_KEYS = {"id", "title", "description", "terminal", "model", "provenance"}
_ROLE_OVERRIDE_KEYS = {"provider", "model", "temperature", "max_tokens"}
_EDGE_KEYS = {
    "id", "source", "target", "prompt_template", "tool_set", "precondition",
    "termination_metric", "produces", "on_fail", "max_attempts", "provenance",
}
_METRIC_KEYS = {"kind", "key", "n", "pattern", "rubric"}
```

applied recursively to top-level graph keys, each node (and its nested
`model` dict when present), and each edge (and its nested `precondition` /
`termination_metric` dicts when present).

**`validate_fn=lambda value: Graph.model_validate(value)`** is passed into
`call_json` (see §4.5) specifically so a structurally-valid-but-semantically-
wrong proposal (e.g. a metric missing its required `n`) is caught as part of
`call_json`'s own retry loop rather than surfacing later as an unhandled
crash. This works because `pydantic.ValidationError` (verified:
`pydantic.ValidationError.__mro__` includes `ValueError`) subclasses
`ValueError`, and `call_json`'s `except (json.JSONDecodeError,
ValidationError, ValueError)` clause already catches `ValueError` — so a
pydantic validation failure inside `validate_fn` is treated exactly like a
JSON-Schema validation failure, triggering the same "return corrected JSON"
retry message.

**Fixed-point pruning loop.** After merging a chapter's nodes and edges, the
loop re-audits and prunes until sound or exhausted:

```python
for _ in range(len(added_nodes) + len(added_edges) + 1):
    result = audit(base)
    if result.ok:
        break
    prunable_nodes = {
        finding.detail.rsplit(": ", 1)[-1]
        for finding in result.findings
        if finding.code in {"dead_end", "unreachable", "cannot_reach_done"}
    } & added_nodes
    if not prunable_nodes:
        break
    base.nodes = [node for node in base.nodes if node.id not in prunable_nodes]
    base.edges = [
        edge for edge in base.edges
        if edge.source not in prunable_nodes and edge.target not in prunable_nodes
    ]
    ...
```

The iteration cap is `len(added_nodes) + len(added_edges) + 1` — enough
rounds to prune every one of this chapter's own additions one at a time plus
one final confirming audit. Only findings coded `dead_end`, `unreachable`, or
`cannot_reach_done` extract an offending node id (via
`finding.detail.rsplit(": ", 1)[-1]`, i.e. the text after the last `": "` in
the finding detail), and that set is intersected with `added_nodes` — **only
this chapter's newly added nodes are ever pruned**, never a seed or
earlier-chapter node. If the intersection is empty (the unsound state can't
be attributed to a prunable node from this chapter) the loop stops early
without forcing soundness. Removing a pruned node also removes every edge
incident to it; an edge added this chapter that is not incident to any
pruned node is never removed by this loop even if it is otherwise part of the
problem.

Two invariants this depends on: it is **safe** because `base` was sound
before this chapter started (the seed graph is sound, and every prior
chapter's merge already ran this same loop to a passing audit before
returning), so any error finding after this chapter's merge must implicate
something this chapter added. Its **limitation** is that only newly added
*nodes* are directly prunable by finding code — a stray added *edge* between
two pre-existing (non-pruned) nodes that itself breaks soundness (e.g. an
edge that creates a cycle bypassing `done`) is only removed as a side effect
of removing one of its endpoint nodes, never removed on its own.

### Event contract

Four event kinds, `_emit` swallowing subscriber exceptions exactly as in
`agent.py` (§4.2):

| event | keys |
|---|---|
| `graph_start` | `event`, `chapters` |
| `graph_chapter_start` | `event`, `chapter`, `index`, `total` |
| `graph_delta` | `event`, `chapter`, `index`, `total`, `new_nodes`, `new_edges`, `node_count`, `edge_count` |
| `graph_complete` | `event`, `audit_ok`, `detail`, `graph` |

`graph_delta` fires once per chapter, immediately after that chapter's merge
and pruning loop settle. `new_nodes` entries are `{"id", "title"}` for every
node currently in `base.nodes` whose id is in that chapter's (post-pruning)
`added_nodes`; `new_edges` entries are `{"id", "source", "target"}` for the
analogous surviving `added_edges`. `node_count`/`edge_count` are the running
totals (`len(base.nodes)` / `len(base.edges)`) after this chapter. This event
streams live to the CLI (`cli/main.py`'s `graph_build` command, which prints
`+X nodes +Y edges (total N/M)` per chapter) and to the TUI (`cli/tui.py`'s
`on_pipeline_msg`, which grows the `#graph` Tree widget per delta).

`graph_complete` fires once, after the loop, with `graph` set to
`base.model_dump(mode="json")` — the full compiled graph, dumped
**regardless of whether it passes audit**. Only after this emit does
`build_graph` raise, if the final audit failed:

```python
result = audit(base)
detail = "; ".join(finding.detail for finding in result.findings if finding.level == "error")
_emit({"event": "graph_complete", "audit_ok": result.ok, "detail": detail, "graph": base.model_dump(mode="json")})
if not result.ok:
    raise GraphError(f"compiled graph failed audit: {detail}")
return base
```

Because the full graph is emitted before the raise, a caller subscribed to
`on_event` can persist the rejected graph (e.g. to `graph.rejected.json`) even
though the function itself raises `GraphError` and never returns it.

### `render.py` — Mermaid rendering

`to_mermaid(graph: Graph) -> str` builds:

1. Header lines: `flowchart TD`, then `%% entry: {graph.entry}`.
2. One line per node, in `graph.nodes` order, two-space indented. The label is
   `_label(node.id, node.title)`: `f"{node_id} — {title}"` with every literal
   `"` replaced by `#quot;`, then every embedded newline collapsed by
   `" ".join(text.split("\n"))` (which also collapses any run of internal
   whitespace, since `str.split()` with no argument splits on any whitespace
   run). Terminal nodes render as a stadium shape,
   `  {id}(["{label}"])`; non-terminal nodes as a rectangle,
   `  {id}["{label}"]`.
3. One line per edge, in `graph.edges` order, **only** if both `edge.source`
   and `edge.target` are present in the node-id set:
   `  {source} -->|{edge_id}| {target}`.
4. The joined lines end with exactly one trailing newline
   (`"\n".join(lines) + "\n"`).

`write_mermaid(graph, path) -> Path` calls
`path.parent.mkdir(parents=True, exist_ok=True)` then
`path.write_text(to_mermaid(graph), encoding="utf-8")`, unconditionally
overwriting any existing file at `path`, and returns `path`.

## 4.5 Structured-output plumbing

### `call_json` (`src/ragent/llm/structured.py`)

Per attempt, `call_json(llm, messages, schema, retries=2, validate_fn=None,
sanitize_fn=None)` runs this pipeline:

1. `llm.complete(current, json_schema=schema)`.
2. `_strip_fences(completion.text)` — strips a leading/trailing Markdown code
   fence (` ``` ` on its own first/last line) if present, else the text is
   used as-is.
3. `json.loads(...)`.
4. **Require a JSON object**: `if not isinstance(value, dict): raise
   ValueError("top-level value must be an object")`.
5. `sanitize_fn(value)` if supplied (e.g. `_sanitize_graph_proposal`).
6. JSON-Schema `validate(instance=value, schema=schema)` (`jsonschema`
   library).
7. `validate_fn(value)` if supplied (e.g. the `Graph.model_validate` lambda in
   §4.4).

`retries + 1` total attempts (default `retries=2` -> 3 attempts). Each retry
does **not** append to the growing conversation; it restarts from the
**original** `messages` list plus exactly one fresh correction message —
the previous (invalid) model response is never replayed back to it:

```python
current = [
    *messages,
    Message(
        "user",
        "Return only corrected JSON matching the supplied schema. "
        f"Previous validation error: {last_error}",
    ),
]
```

`last_error` is `str(exc)` from whichever of `json.JSONDecodeError`,
`jsonschema.ValidationError`, or `ValueError` (which also catches
`pydantic.ValidationError`, per §4.4) was raised. Exhausting all attempts
raises `ProviderError`:

```
{llm.role} provider returned invalid structured output after {retries + 1} attempts: {last_error}
```

### OpenRouter-specific payload additions (`src/ragent/llm/openai_compat.py`)

`OpenAICompatProvider` is constructed with a `report_cost: bool = False` flag;
`ragent`'s provider factory sets this `True` only for `kind == "openrouter"`.
Two payload additions are gated on it inside `complete()`:

- **`payload["usage"] = {"include": True}`** — sent on every completion when
  `report_cost` is true. This is what makes OpenRouter the only configured
  provider that reports real USD cost per call (`data.get("usage",
  {}).get("cost")`), which in turn is what lets a `cost_limit_usd` budget be
  enforced meaningfully (see [algorithm.md](algorithm.md) §1.5) — other
  providers never populate `cost`, so their spend only counts against a
  budget if the workspace config supplies a manual `[prices."provider/model"]`
  entry.
- **`payload["reasoning"] = {"max_tokens": max(256, max_tokens // 4)}`** —
  added only when `json_schema is not None` (a structured/schema call) *and*
  `report_cost` is true. The source comment explains why: OpenRouter
  reasoning models can spend the entire completion budget on chain-of-thought
  before ever emitting the JSON payload, returning empty content; capping the
  reasoning budget to a quarter of `max_tokens` (floor 256) leaves room for
  the actual payload. The same comment notes some models reject an explicit
  `enabled: false` ("mandatory reasoning"), which is why the code bounds the
  reasoning budget instead of trying to disable it outright.

### 400-fallback retry

If the initial POST raises a `ProviderError` whose message contains
`"400"` **and** the call was a schema call (`json_schema is not None`), the
request is retried exactly once with `response_format` removed and one
appended user message:

```python
payload.pop("response_format", None)
payload["messages"].append({
    "role": "user",
    "content": "Return only JSON matching this schema: " + json.dumps(json_schema),
})
data = self._post(payload)
```

Any other error (non-400, or a non-schema call) is re-raised immediately
without this fallback.
