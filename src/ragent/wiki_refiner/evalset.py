from __future__ import annotations

import json
from importlib.resources import files


def load_eval_set() -> list[str]:
    return list(json.loads(files("ragent.data").joinpath("eval_set.json").read_text(encoding="utf-8")))
