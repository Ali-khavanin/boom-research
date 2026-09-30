# ragent docs

`ragent` delegates PDF-to-skill conversion to upstream `book-to-skill` through a
host agent, then compiles upstream `chapters/ch<NN>-*.md` into an audited,
metric-gated stage graph. Research queries walk that graph, recording each step.

| Page | Read this when… |
|---|---|
| [algorithm](algorithm.md) | you want to know exactly how a run executes: edge selection, retries, tool loop, metrics, budget enforcement. |
| [skill-graph](skill-graph.md) | you want to understand what a "skill graph" is, the shipped seed graph, its schema, and its soundness rules. |
| [trajectories](trajectories.md) | you want to know what gets recorded per run, how to export/aggregate it, and how offline graph refinement works. |
| [book-to-skill](book-to-skill.md) | you want to install the upstream host skill, configure conversion, validate its artifacts, and compile chapters into the graph. |

Pipeline order:

```
ragent book <pdf>  →  ragent graph build  →  ragent research "<query>"  →  ragent runs export  →  ragent refine propose
```

See the root [README](../README.md) for installation, configuration, and the full CLI/TUI reference.
