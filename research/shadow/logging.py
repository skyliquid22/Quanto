"""Append-only logging for shadow execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class ShadowLogger:
    """JSONL writer capturing per-step execution records."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir = self.base_dir / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._steps_path = self._logs_dir / "steps.jsonl"
        self._summary_path = self.base_dir / "summary.json"

    @property
    def steps_path(self) -> Path:
        return self._steps_path

    @property
    def summary_path(self) -> Path:
        return self._summary_path

    def append(self, record: Mapping[str, object]) -> None:
        serialized = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._steps_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")

    def write_summary(self, payload: Mapping[str, object]) -> Path:
        ordered = {key: payload[key] for key in sorted(payload)}
        self._summary_path.write_text(json.dumps(ordered, sort_keys=True, indent=2), encoding="utf-8")
        return self._summary_path


__all__ = ["ShadowLogger"]
