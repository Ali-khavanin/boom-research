# ragent docs

`ragent` compiles a research methodology (a PDF book or paper) into an audited, metric-gated stage
graph, then runs research queries through that graph, recording every step as a trajectory.

| Page | Read this when… |
|---|---|
| [algorithm](algorithm.md) | you want to know exactly how a run executes: edge selection, retries, tool loop, metrics, budget enforcement. |
| [skill-graph](skill-graph.md) | you want to understand what a "skill graph" is, the shipped seed graph, its schema, and its soundness rules. |
| [trajectories](trajectories.md) | you want to know what gets recorded per run, how to export/aggregate it, and how offline graph refinement works. |
| [book-to-skill](book-to-skill.md) | you want to know how a PDF becomes skill chapters and how chapters are merged into the graph. |

Pipeline order:

```
ragent book <pdf>  →  ragent graph build  →  ragent research "<query>"  →  ragent runs export  →  ragent refine propose
```

See the root [README](../README.md) for installation, configuration, and the full CLI/TUI reference.
