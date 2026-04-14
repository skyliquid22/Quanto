"""Output dataclasses for stress test runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class GateResult:
    """Outcome of evaluating a single gate against a scenario run."""

    gate_name: str
    status: Literal["pass", "warn", "fail", "skipped"]
    measured: dict[str, float]
    thresholds: dict[str, float]
    reason: str | None = None


@dataclass
class ScenarioRunResult:
    """Outcome of a single scenario × seed execution."""

    scenario_name: str
    seed: int
    status: Literal["pass", "warn", "fail", "error"]
    gate_results: list[GateResult] = field(default_factory=list)
    run_dir: Path | None = None
    error: str | None = None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "seed": self.seed,
            "status": self.status,
            "error": self.error,
            "gates": [
                {
                    "gate": g.gate_name,
                    "status": g.status,
                    "measured": g.measured,
                    "thresholds": g.thresholds,
                    "reason": g.reason,
                }
                for g in self.gate_results
            ],
        }


@dataclass
class StressTestResult:
    """Aggregated result across all scenarios and seeds."""

    config_name: str
    overall_status: Literal["pass", "fail"]
    scenario_results: list[ScenarioRunResult] = field(default_factory=list)
    n_pass: int = 0
    n_warn: int = 0
    n_fail: int = 0
    n_error: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "overall_status": self.overall_status,
            "counts": {
                "pass": self.n_pass,
                "warn": self.n_warn,
                "fail": self.n_fail,
                "error": self.n_error,
                "total": self.n_pass + self.n_warn + self.n_fail + self.n_error,
            },
            "runs": [r.summary_dict() for r in self.scenario_results],
        }


def aggregate_results(scenario_results: list[ScenarioRunResult], config_name: str) -> StressTestResult:
    """Combine per-run results into a single StressTestResult."""
    counts: dict[str, int] = {"pass": 0, "warn": 0, "fail": 0, "error": 0}
    for result in scenario_results:
        counts[result.status] = counts.get(result.status, 0) + 1
    overall = "fail" if (counts["fail"] + counts["error"]) > 0 else "pass"
    return StressTestResult(
        config_name=config_name,
        overall_status=overall,
        scenario_results=scenario_results,
        n_pass=counts["pass"],
        n_warn=counts["warn"],
        n_fail=counts["fail"],
        n_error=counts["error"],
    )


__all__ = [
    "GateResult",
    "ScenarioRunResult",
    "StressTestResult",
    "aggregate_results",
]
