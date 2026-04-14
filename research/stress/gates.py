"""Portfolio-level gate evaluators for stress test scenarios.

These gates operate on the output of a completed ShadowEngine run (steps.jsonl)
and assess portfolio-level outcomes: drawdown, VaR, and liquidity. They are
distinct from ExecutionGateRunner, which checks execution-layer metrics like
order reject rate and broker halts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from research.stress.config import GateConfig
from research.stress.results import GateResult


def _load_steps(steps_path: Path) -> list[dict[str, Any]]:
    """Load all records from a steps.jsonl file."""
    if not steps_path.exists():
        return []
    records = []
    for line in steps_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _portfolio_values(steps: list[dict[str, Any]]) -> list[float]:
    return [float(s["portfolio_value"]) for s in steps if "portfolio_value" in s]


def _daily_returns(values: list[float]) -> list[float]:
    if len(values) < 2:
        return []
    return [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values)) if values[i - 1] != 0.0]


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = float(values[0])
    worst = 0.0
    for v in values:
        fv = float(v)
        if fv > peak:
            peak = fv
        dd = (fv - peak) / peak if peak != 0.0 else 0.0
        worst = min(worst, dd)
    return abs(worst)


def _historical_var(returns: list[float], confidence: float) -> float:
    """One-day historical VaR at the given confidence level (e.g. 0.95).

    Returns a positive number representing the loss threshold: a VaR of 0.03
    means there is a (1-confidence) probability of losing more than 3% in a day.
    """
    if not returns:
        return 0.0
    sorted_returns = sorted(returns)
    index = int(math.floor((1.0 - confidence) * len(sorted_returns)))
    index = max(0, min(index, len(sorted_returns) - 1))
    return abs(sorted_returns[index])


def evaluate_drawdown_gate(steps: list[dict[str, Any]], config: GateConfig) -> GateResult:
    values = _portfolio_values(steps)
    mdd = _max_drawdown(values)
    max_dd_threshold = float(config.thresholds.get("max_drawdown", 1.0))
    warn_threshold = float(config.thresholds.get("warning_threshold", max_dd_threshold))
    measured = {"max_drawdown": round(mdd, 6)}
    thresholds = {"max_drawdown": max_dd_threshold, "warning_threshold": warn_threshold}
    if mdd >= max_dd_threshold:
        return GateResult(
            "drawdown",
            "fail",
            measured,
            thresholds,
            f"max_drawdown {mdd:.4f} >= threshold {max_dd_threshold:.4f}",
        )
    if mdd >= warn_threshold:
        return GateResult(
            "drawdown",
            "warn",
            measured,
            thresholds,
            f"max_drawdown {mdd:.4f} >= warning_threshold {warn_threshold:.4f}",
        )
    return GateResult("drawdown", "pass", measured, thresholds)


def evaluate_var_gate(steps: list[dict[str, Any]], config: GateConfig) -> GateResult:
    values = _portfolio_values(steps)
    returns = _daily_returns(values)
    var95 = _historical_var(returns, 0.95)
    var99 = _historical_var(returns, 0.99)
    var95_threshold = float(config.thresholds.get("var_95", 1.0))
    var99_threshold = float(config.thresholds.get("var_99", 1.0))
    measured = {"var_95": round(var95, 6), "var_99": round(var99, 6)}
    thresholds = {"var_95": var95_threshold, "var_99": var99_threshold}
    failures = []
    if var95 > var95_threshold:
        failures.append(f"VaR(95%) {var95:.4f} > {var95_threshold:.4f}")
    if var99 > var99_threshold:
        failures.append(f"VaR(99%) {var99:.4f} > {var99_threshold:.4f}")
    if failures:
        return GateResult("var_limit", "fail", measured, thresholds, "; ".join(failures))
    return GateResult("var_limit", "pass", measured, thresholds)


def evaluate_liquidity_gate(steps: list[dict[str, Any]], config: GateConfig) -> GateResult:
    """Check that average daily fill volume meets the minimum threshold.

    Reads fill notional from execution.fills in each step record. When no fill
    data is present (e.g. execution_mode="none") the gate passes vacuously.
    """
    min_volume = float(config.thresholds.get("min_daily_volume", 0.0))
    fill_volumes: list[float] = []
    for step in steps:
        fills = step.get("execution", {}).get("fills", [])
        for fill in fills:
            notional = float(fill.get("notional", 0.0))
            fill_volumes.append(abs(notional))
    if not fill_volumes:
        return GateResult(
            "liquidity",
            "pass",
            {"avg_daily_volume": 0.0},
            {"min_daily_volume": min_volume},
            "no fill data — gate passed vacuously",
        )
    avg_volume = sum(fill_volumes) / len(fill_volumes)
    measured = {"avg_daily_volume": round(avg_volume, 2)}
    thresholds = {"min_daily_volume": min_volume}
    if avg_volume < min_volume:
        return GateResult(
            "liquidity",
            "fail",
            measured,
            thresholds,
            f"avg_daily_volume {avg_volume:.2f} < min_daily_volume {min_volume:.2f}",
        )
    return GateResult("liquidity", "pass", measured, thresholds)


_GATE_HANDLERS = {
    "drawdown": evaluate_drawdown_gate,
    "var_limit": evaluate_var_gate,
    "liquidity": evaluate_liquidity_gate,
}


class StressGateEvaluator:
    """Evaluates all configured gates against a completed scenario run."""

    def __init__(self, gates: dict[str, GateConfig]) -> None:
        self._gates = gates

    def evaluate(self, run_dir: Path) -> list[GateResult]:
        """Load steps.jsonl from run_dir and evaluate all gates."""
        steps_path = run_dir / "logs" / "steps.jsonl"
        steps = _load_steps(steps_path)
        results = []
        for gate_name, gate_config in self._gates.items():
            if not gate_config.enabled:
                results.append(
                    GateResult(gate_name, "skipped", {}, dict(gate_config.thresholds), "gate disabled")
                )
                continue
            handler = _GATE_HANDLERS.get(gate_name)
            if handler is None:
                results.append(
                    GateResult(gate_name, "skipped", {}, dict(gate_config.thresholds), f"unknown gate '{gate_name}'")
                )
                continue
            results.append(handler(steps, gate_config))
        return results


__all__ = [
    "StressGateEvaluator",
    "evaluate_drawdown_gate",
    "evaluate_var_gate",
    "evaluate_liquidity_gate",
]
