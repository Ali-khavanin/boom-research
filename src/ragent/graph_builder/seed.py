from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from .schema import Graph


def load_seed(path: Path | None = None) -> Graph:
    if path is None:
        data = json.loads(files("ragent.data").joinpath("seed_graph.json").read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    return Graph.model_validate(data)
