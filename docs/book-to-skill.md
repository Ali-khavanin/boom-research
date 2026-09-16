# Document to Skill Graph

This page documents the four-stage pipeline that turns a methodology PDF into an
executable stage graph: PDF text extraction, chapter segmentation, LLM
distillation into skill chapters plus `SKILL.md`, and the merge of those
chapters into `graph.json`. See [algorithm.md](algorithm.md) for how the
resulting graph is executed and [skill-graph.md](skill-graph.md) for the graph
schema and soundness rules.

## 4.1 Text extraction

`src/ragent/book_to_skill/pdf.py` is a thin adapter over the pinned upstream
`book-to-skill` package (`book-to-skill[pdf] @ git+...@v1.4.0`, installed as
the `book_to_skill` distribution). The entire module:

```python
from book_to_skill import ExtractionError, extract_single_file
...
def extract(pdf: Path, mode: str = "text") -> ExtractedBook:
    try:
        result = extract_single_file(
            pdf.expanduser().resolve(),
            extraction_mode=mode,
            install_mode="no",
        )
    except ExtractionError as exc:
        raise RagentError(str(exc)) from exc
    text = str(result.pop("text", "")).strip()
    if not text:
        raise RagentError(
            f"no extractable text in {pdf}; run OCR on the scanned PDF first"
        )
    return ExtractedBook(
        text=text,
        pages=int(result.get("pages") or 0),
        metadata=result,
    )
```

`extract_single_file` is called with the resolved absolute path,
`extraction_mode=mode` (`ragent`'s `extract()` defaults `mode="text"`), and
`install_mode="no"` — upstream is never allowed to auto-install missing
system tools. Any `book_to_skill.ExtractionError` is caught and re-raised as
`ragent.errors.RagentError` with the upstream message preserved verbatim
(`str(exc)`). If upstream returns text that is empty after stripping, `pdf.py`
raises its own error, exactly:

```
no extractable text in {pdf}; run OCR on the scanned PDF first
```

`ExtractedBook` (`dataclass(slots=True)`) has three fields: `text: str`,
`pages: int`, `metadata: dict[str, Any]`. `metadata` is upstream's full result
dict with the `"text"` key popped out — every other key upstream returns is
passed through unchanged. Reading `extract_single_file` in the installed
package (`book_to_skill/utils.py`), the dict it builds for any input format is:

```python
{
    "source_file": str(input_path.resolve()),
    "filename": input_path.name,
    "format": document_format,
    "extraction_method": method,
    "file_size_mb": round(file_size_mb, 2),
    pages_label: pages,        # e.g. "pages": N for PDFs
    "pages_label": pages_label,
    "pages": pages,
    "chars": len(text),
    "words": len(text.split()),
    "estimated_tokens": tokens,
    "text": text,
    **structure,                # chapters_detected, chapter_headings_sample, has_toc
}
```

So after `pdf.py` pops `"text"`, `ExtractedBook.metadata` for a PDF carries
exactly: `source_file`, `filename`, `format`, `extraction_method`,
`file_size_mb`, `pages_label` (the string `"pages"` for PDFs), `pages`,
`chars`, `words`, `estimated_tokens`, `chapters_detected`,
`chapter_headings_sample`, `has_toc`. Running `extract()` on the bundled
`08IJBAS31.pdf` in this repo produced exactly this key set (verified this
session; see §4.2 for the full worked run).

The pdftotext -> pypdf -> pdfminer.six fallback chain, the image-only-PDF
refusal, and the printed progress lines all belong to the pinned upstream
package (`book_to_skill/utils.py` in the installed `book-to-skill` v1.4.0
distribution), not to this repo's code — they are documented here as observed
external behavior. For a PDF, upstream prints `Extracting PDF: <path>`, then
in text mode tries each extractor in order, printing `Trying pdftotext... `
followed by `OK` or `not available`, then (on failure) `Trying pypdf... `,
then `Trying pdfminer.six... `, ending in `FAILED` plus an install-one-of
message if none succeed. Before extraction it checks `looks_image_only()` and
if true raises an `ExtractionError` whose message ends with:

```
Run OCR on it first, then retry:
  ocrmypdf input.pdf output.pdf
```

Running `extract()` locally against `08IJBAS31.pdf` in this environment
(`pdftotext` binary absent, `pypdf` present) actually printed:

```
Extracting PDF: .../08IJBAS31.pdf
Mode: text — using pdftotext...
Trying pdftotext... not available
Trying pypdf... OK
```

with `extraction_method: "pypdf"` in the resulting metadata.

## 4.2 Segmentation

`src/ragent/book_to_skill/chapters.py`'s `segment(book: ExtractedBook) ->
list[Chapter]` tries three strategies in this precedence order, falling
through only when the higher-precedence strategy yields nothing usable.

### Branch 1 — keyword headings

```python
_CHAPTER = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+)?"
    r"((?:chapter|unit|lesson|module|lecture|part|section)\s+"
    r"(?:\d{1,3}|[IVXLCDM]{1,7})\b[^\n]*)"
)
```

This matches, at line start (case-insensitive, multiline), an optional
Markdown `#` prefix, then one of the keywords `chapter`, `unit`, `lesson`,
`module`, `lecture`, `part`, `section` followed by an Arabic (`\d{1,3}`) or
Roman (`[IVXLCDM]{1,7}`) numeral and the rest of that line. `segment()` uses
this branch only when `len(matches) >= 2`; a single keyword hit is not enough
to trust as chapter structure.

### Branch 2 — numbered sections

```python
_NUMBERED = re.compile(r"(?m)^[ \t]*(\d{1,2})[.)]?[ \t]+([A-Z][^\n]{2,80}?)[ \t]*$")
```

`_numbered_sections()` (module docstring: *"Longest ascending 1,2,3… run of
numbered headings"*) runs, in order:

1. **Collect** every regex match as `(number, start_offset, title)`, dropping
   any `number > 20`.
2. **Global uniqueness filter** — normalize each title
   (`" ".join(title.lower().split())`), count occurrences across all
   candidates, and keep only candidates whose normalized title occurs exactly
   once. This removes repeated running headers/footers (a number that recurs
   with the same title on many pages is not a real section boundary).
3. **Build source-order consecutive runs** — walk the filtered candidates in
   the order the regex found them; a candidate extends the current run only
   if its number is exactly one more than the run's last number, otherwise it
   starts a new run.
4. **Qualify** — keep only runs whose first number is `1` and whose length is
   `>= 3` (`qualifying = [run for run in runs if run[0][0] == 1 and len(run) >= 3]`).
   No qualifying run -> the branch returns `[]` and `segment()` falls through
   to block splitting.
5. **Pick the winner** — `qualifying.sort(key=lambda run: run[0][1])` sorts
   qualifying runs by the character offset of their first heading, then
   `return qualifying[-1]` returns the **last** one in that order.

**Verified discrepancy:** step 5 selects the run whose `1` starts **latest**
in the document, not the longest run — despite the function's own docstring
and inline comment (`"Longest ascending ... run"`) claiming length wins. Any
qualifying run tied for "starts last" beats a qualifying run that is merely
longer but starts earlier. This documents the code's actual behavior, which
is what `ragent` runs; the docstring is stale. Practically this means a short
`1./2./3.` author-affiliation or acknowledgments block that happens to sit
later in the source text can still lose to an earlier, longer body-section
run only if the body run itself starts later — and conversely a late, short
qualifying run (e.g., trailing numbered appendix items) can beat an earlier,
longer body-section run.

### Branch 3 — fixed-size blocks (fallback)

```python
block_count = detected or max(1, math.ceil((book.pages or 12) / 12))
block_size = max(1, math.ceil(len(book.text) / block_count))
```

where `detected = int(book.metadata.get("chapters_detected") or 0)`. Chapters
are titled `Section 1`, `Section 2`, … and text is sliced into `block_size`
windows in character order.

### Front Matter prepending

Both the keyword-heading and numbered-section branches prepend a synthetic
`Front Matter` chapter — `text[: matches[0].start()]` / `text[: sections[0][1]]`
— only when the first detected heading starts **beyond character offset
500**. This skips the synthetic chapter when the document's real structure
begins immediately (no meaningful preamble to capture) and only pages
title/author/abstract text into its own chapter when there is a nontrivial
prefix in front of the first heading. Empty resulting chapter bodies (after
`.strip()`) are skipped in every branch (`if text:` guards); the fallback
block branch has no such check since blocks are lengths, not delimiter-based.

### `Chapter` and page-range estimation

```python
@dataclass(slots=True)
class Chapter:
    index: int
    title: str
    text: str
    pages: list[int]
```

`_page_range(book, start, end)` returns `[]` if `book.pages <= 0` or the text
is empty; otherwise it estimates a page range purely from the **proportion of
character offset within the extracted text**, not from any real per-page
provenance the extractor tracked:

```python
first = max(1, math.floor(start / len(book.text) * book.pages) + 1)
last = min(book.pages, max(first, math.ceil(end / len(book.text) * book.pages)))
return list(range(first, last + 1))
```

Because upstream's extractors (pdftotext/pypdf/pdfminer) do not preserve
page-boundary markers in the returned plain text, this is the only page
estimate available; it is linear-interpolated, not extracted.

### Worked example (`08IJBAS31.pdf`, bundled in the repo root)

Running `ragent.book_to_skill.pdf.extract()` then
`ragent.book_to_skill.chapters.segment()` locally against the bundled PDF
(pure text extraction and regex — no LLM call, no network) in this
environment produced the numbered-section branch's winning run (`pages: 10`,
`chapters_detected: 0`, so branch 1 did not fire with >=2 matches and branch 3
never ran):

| index | title | chars | estimated pages |
|---|---|---|---|
| 1 | `Front Matter` | 1483 | `[1]` |
| 2 | `1 Introduction` | 2096 | `[1, 2]` |
| 3 | `2 What is a Literature Review` | 3440 | `[2, 3]` |
| 4 | `3 Systematic Literature Review` | 2042 | `[3]` |
| 5 | `4 Steps in the Literature Review Process` | 17776 | `[3, 4, 5, 6, 7, 8, 9]` |
| 6 | `5 Conclusion` | 3461 | `[9, 10]` |

`extraction_method` was `pypdf` in this run (`pdftotext` binary not present
on this machine; `pypdf` succeeded). This is one run's output, not a
guaranteed-stable fixture — re-running against a different-PDF fixture will
segment differently, but the branch precedence and selection rules above are
exact.

## 4.3 Chapter distillation and `SKILL.md`

`src/ragent/book_to_skill/writer.py`'s
`build(pdf, out_dir, llm, force=False, on_event=None) -> SkillBundle` drives
one LLM call per chapter plus one summary call.

### Chapter prompt (verbatim)

System message:

```
Convert research-methodology source text into an executable skill chapter. Do not invent methods or evidence.
```

User message template (`f"..."`, `chapter.title`/`chapter.pages` interpolated,
followed by the truncated chapter text):

```
Write Markdown with these sections: Purpose, When to use, Ordered procedure, Decision rules, and Discourse cues. Preserve relevant cue words verbatim, especially first, then, before, if, unless, in order to, so that, and until.
Source title: {chapter.title}
Source pages: {chapter.pages}

{truncated chapter text}
```

### SKILL.md prompt (verbatim)

System message:

```
Synthesize chapter skills into one faithful end-to-end research procedure. Keep stage order, decision rules, loops, and source chapter links explicit.
```

User message (chapter bodies joined with a blank line, appended after the
instruction):

```
Produce the body of SKILL.md in Markdown. Include Overview, End-to-end procedure, Decision points, and Chapters. Do not add YAML front matter.

{rendered chapter bodies joined by "\n\n"}
```

### Cue words

```python
_CUES = ("first", "then", "before", "if", "unless", "in order to", "so that", "until")
```

`_stage_hints(chapter)` lowercases the chapter text once and returns the
subset of `_CUES` present as a substring (`[cue for cue in _CUES if cue in
lowered]`), preserving `_CUES` order — this list becomes each chapter's
`stage_hints` front-matter value.

### File paths and slugs

Chapter files are written to `chapters/{chapter.index:02d}-{_slug(chapter.title)}.md`.
`_slug`:

```python
def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:72] or "section"
```

— lowercase, non-alphanumeric runs collapsed to a single `-`, leading/trailing
`-` stripped, truncated to 72 characters, falling back to the literal
`"section"` if the result is empty.

### Front matter

Every chapter file is written as:

```
---
name: <json.dumps(chapter.title)>
source_pages: <json.dumps(chapter.pages)>
stage_hints: <json.dumps(_stage_hints(chapter))>
---

{llm chapter content}
```

All three values are `json.dumps`-encoded (so a title containing a quote or
Unicode is a valid inline JSON scalar within the YAML block, not raw text).
`SKILL.md`'s front matter is analogous but with different keys — `name`
(`json.dumps(pdf.stem)`), `description`
(`json.dumps("Research methodology extracted from " + pdf.name)`), and
`chapters` (`json.dumps([str(path.relative_to(out_dir)) for path in paths])`,
i.e. the list of chapter-relative paths, encoded once as a single JSON array
literal).

### Truncation

```python
def _truncate(text: str) -> str:
    if len(text) <= 24_000:
        return text
    return text[:16_000] + "\n\n[...middle omitted...]\n\n" + text[-8_000:]
```

Chapter text at or under 24,000 characters is sent to the LLM unchanged;
longer text is cut to its first 16,000 and last 8,000 characters with a
`[...middle omitted...]` marker in between. Only the chapter-generation
prompt truncates; the SKILL.md prompt concatenates already-generated
(LLM-authored, therefore already-bounded) chapter bodies without further
truncation.

### Cache semantics

Per chapter: `cached = path.exists() and not force`. If cached, the file is
read verbatim from disk with **no freshness check** against the source PDF or
chapter text — a stale chapter file is trusted as-is until `force=True`.
`SKILL.md` has an **independent** cache flag, `skill_cached =
skill_path.exists() and not force`, computed once after the entire chapter
loop and checked before the summary call. Because it does not depend on
whether any individual chapter was regenerated this run, regenerating one
chapter (e.g. by deleting just that chapter's file) leaves a stale
`SKILL.md` untouched unless `--force` is also passed — `--force` is therefore
the only way to guarantee both chapters and `SKILL.md` are rebuilt together.

### `on_event` payloads

Read directly from the five `_emit(...)` call sites in `build()`:

| event | keys |
|---|---|
| `book_start` | `event`, `pdf`, `chapters` |
| `chapter_start` | `event`, `index`, `total`, `title`, `chars`, `cached` |
| `chapter_done` | `event`, `index`, `total`, `title`, `path`, `cached` |
| `skill_start` | `event`, `cached` |
| `skill_done` | `event`, `path` |

`_emit` wraps every call to `on_event` in a bare `try/except Exception: pass`,
so a raising subscriber (a broken progress UI) is silently swallowed and
never aborts the underlying (potentially paid) LLM run:

```python
def _emit(payload: dict[str, Any]) -> None:
    if on_event is None:
        return
    try:
        on_event(payload)
    except Exception:
        pass
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

Chapter files come from `sorted((skill_dir / "chapters").glob("*.md"))` (empty
list if the directory does not exist), so chapters are always processed in
filename order — the same `NN-slug.md` order `writer.py` wrote them in.

### Per-chapter graph-compilation prompt (verbatim)

System message:

```
Compile a methodology chapter into a sound research-stage graph. Reuse the seed ids start, goal, what_has_been_done, limitations, gaps, feasibility, quick_test, done whenever applicable. Mine first/then/next/after as sequencing; if/unless/when as preconditions or loop-backs; in order to/so that as prompt intent. Add a node only for a genuinely new stage. Every non-seed node and edge needs provenance with chapter and the exact cue. Metric field rules: kinds min_words, min_items, and has_citations each require an integer n (e.g. has_citations needs n = minimum citation count); kind regex requires pattern; kind llm_rubric requires rubric. Connectivity is mandatory: your JSON is merged as-is, with no cross-chapter wiring added afterward. Every node you include (seed or new) must sit on at least one edge path that starts at a seed node reachable from 'start' and ends at 'done'. Never add a node with no outgoing edge unless it is 'done' itself; never add a node with no incoming edge from 'start' or another node already on such a path. If a new node does not chain forward to 'done', omit it rather than leave it disconnected.
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
`writer.py` (§4.3):

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
