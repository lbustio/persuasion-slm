from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el benchmark: {path}")

    cases: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["_line_no"] = line_no
            cases.append(payload)
    if not cases:
        raise ValueError(f"El benchmark {path} no contiene casos.")
    return cases


def default_benchmark_path() -> Path:
    return Path(__file__).resolve().parents[2] / "intelligence_eval" / "benchmarks" / "seed_cases.jsonl"

