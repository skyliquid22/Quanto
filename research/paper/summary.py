"""Daily artifact writers for paper execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class DailySummaryWriter:
    """Persists human-auditable artifacts for paper runs."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, date_key: str, payload: Mapping[str, object]) -> tuple[Path, Path | None]:
        required = ["pnl", "exposure", "turnover", "fees", "halt_reasons"]
        for field in required:
            if field not in payload:
                raise ValueError(f"Daily summary missing required field '{field}'")
        json_path = self._run_dir / f"daily_summary_{date_key}.json"
        json_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        md_path = self._run_dir / f"daily_summary_{date_key}.md"
        lines = [f"# Paper Run Summary {date_key}", ""]
        for key, value in payload.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return json_path, md_path


@dataclass
class GateThresholds:
    """Absolute thresholds for execution gates."""

    max_reject_rate: float = 0.25
    max_hard_drawdown: float = 0.1


class ExecutionGateRunner:
    """Evaluates execution regression gates for paper runs."""

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self._thresholds = thresholds or GateThresholds()

    def evaluate(self, metrics: Mapping[str, object]) -> dict[str, object]:
        summary = metrics.get("summary") if isinstance(metrics, Mapping) else None
        if not isinstance(summary, Mapping):
            return {"status": "UNKNOWN", "hard_failures": ["missing_metrics"], "soft_failures": []}
        reject_rate = float(summary.get("reject_rate", 0.0))
        drawdown = float(summary.get("execution_halts", 0))
        hard_failures: list[str] = []
        soft_failures: list[str] = []
        if reject_rate > self._thresholds.max_reject_rate:
            hard_failures.append(f"reject_rate>{self._thresholds.max_reject_rate}")
        if drawdown > self._thresholds.max_hard_drawdown:
            soft_failures.append(f"halt_events>{self._thresholds.max_hard_drawdown}")
        status = "HALTED" if hard_failures else "OK"
        return {"status": status, "hard_failures": hard_failures, "soft_failures": soft_failures}


__all__ = ["DailySummaryWriter", "ExecutionGateRunner", "GateThresholds"]
