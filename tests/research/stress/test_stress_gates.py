"""Unit tests for stress test gate evaluators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.stress.config import GateConfig
from research.stress.gates import (
    StressGateEvaluator,
    evaluate_drawdown_gate,
    evaluate_var_gate,
    evaluate_liquidity_gate,
    _load_steps,
    _max_drawdown,
    _historical_var,
    _daily_returns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_steps(portfolio_values: list[float]) -> list[dict]:
    return [{"portfolio_value": v} for v in portfolio_values]


def _write_steps(tmp_path: Path, steps: list[dict]) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / "steps.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in steps) + "\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_flat():
    assert _max_drawdown([100.0, 100.0, 100.0]) == pytest.approx(0.0)


def test_max_drawdown_monotone_up():
    assert _max_drawdown([100.0, 110.0, 120.0]) == pytest.approx(0.0)


def test_max_drawdown_single_drop():
    # Peak 100 → trough 80 → 20% drawdown
    assert _max_drawdown([100.0, 90.0, 80.0]) == pytest.approx(0.2)


def test_max_drawdown_recovery():
    # 100 → 80 → 100 — still records 20% peak-to-trough
    assert _max_drawdown([100.0, 80.0, 100.0]) == pytest.approx(0.2)


def test_max_drawdown_empty():
    assert _max_drawdown([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _historical_var
# ---------------------------------------------------------------------------

def test_var_empty():
    assert _historical_var([], 0.95) == pytest.approx(0.0)


def test_var_positive_returns_only():
    # All positive returns — VaR is the smallest positive return (abs)
    returns = [0.01, 0.02, 0.03, 0.04, 0.05]
    var = _historical_var(returns, 0.95)
    assert var >= 0.0


def test_var_large_loss_in_tail():
    # 5 returns: 5% tail = index 0 = the largest loss
    returns = [0.01, 0.02, 0.01, 0.01, -0.20]
    var95 = _historical_var(returns, 0.95)
    assert var95 == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# drawdown gate
# ---------------------------------------------------------------------------

def test_drawdown_gate_pass():
    steps = _make_steps([100.0, 102.0, 104.0, 103.0])
    config = GateConfig(enabled=True, thresholds={"max_drawdown": 0.15, "warning_threshold": 0.10})
    result = evaluate_drawdown_gate(steps, config)
    assert result.status == "pass"
    assert result.measured["max_drawdown"] < 0.10


def test_drawdown_gate_warn():
    # 100 → 88 = 12% drawdown, above warning (10%) but below fail (15%)
    steps = _make_steps([100.0, 95.0, 88.0, 90.0])
    config = GateConfig(enabled=True, thresholds={"max_drawdown": 0.15, "warning_threshold": 0.10})
    result = evaluate_drawdown_gate(steps, config)
    assert result.status == "warn"


def test_drawdown_gate_fail():
    # 100 → 80 = 20% drawdown, above fail threshold (15%)
    steps = _make_steps([100.0, 90.0, 80.0])
    config = GateConfig(enabled=True, thresholds={"max_drawdown": 0.15, "warning_threshold": 0.10})
    result = evaluate_drawdown_gate(steps, config)
    assert result.status == "fail"
    assert "max_drawdown" in result.reason


def test_drawdown_gate_empty_steps():
    config = GateConfig(enabled=True, thresholds={"max_drawdown": 0.15})
    result = evaluate_drawdown_gate([], config)
    assert result.status == "pass"
    assert result.measured["max_drawdown"] == 0.0


# ---------------------------------------------------------------------------
# VaR gate
# ---------------------------------------------------------------------------

def test_var_gate_pass():
    # Tiny daily moves — VaR well under thresholds
    values = [100.0 + i * 0.1 for i in range(50)]
    steps = _make_steps(values)
    config = GateConfig(enabled=True, thresholds={"var_95": 0.05, "var_99": 0.08})
    result = evaluate_var_gate(steps, config)
    assert result.status == "pass"


def test_var_gate_fail():
    # Spike a large single-day loss into the tail
    values = [100.0] * 20 + [70.0] + [100.0] * 79  # 30% loss in one day
    steps = _make_steps(values)
    config = GateConfig(enabled=True, thresholds={"var_95": 0.05, "var_99": 0.08})
    result = evaluate_var_gate(steps, config)
    assert result.status == "fail"


def test_var_gate_empty_steps():
    config = GateConfig(enabled=True, thresholds={"var_95": 0.05, "var_99": 0.08})
    result = evaluate_var_gate([], config)
    assert result.status == "pass"


# ---------------------------------------------------------------------------
# Liquidity gate
# ---------------------------------------------------------------------------

def test_liquidity_gate_no_fills_passes_vacuously():
    steps = [{"portfolio_value": 100.0, "execution": {"fills": []}}]
    config = GateConfig(enabled=True, thresholds={"min_daily_volume": 1_000_000})
    result = evaluate_liquidity_gate(steps, config)
    assert result.status == "pass"
    assert "vacuously" in (result.reason or "")


def test_liquidity_gate_pass():
    fills = [{"notional": 2_000_000.0}]
    steps = [{"portfolio_value": 100.0, "execution": {"fills": fills}}]
    config = GateConfig(enabled=True, thresholds={"min_daily_volume": 1_000_000})
    result = evaluate_liquidity_gate(steps, config)
    assert result.status == "pass"


def test_liquidity_gate_fail():
    fills = [{"notional": 100.0}]
    steps = [{"portfolio_value": 100.0, "execution": {"fills": fills}}]
    config = GateConfig(enabled=True, thresholds={"min_daily_volume": 1_000_000})
    result = evaluate_liquidity_gate(steps, config)
    assert result.status == "fail"


# ---------------------------------------------------------------------------
# StressGateEvaluator
# ---------------------------------------------------------------------------

def test_evaluator_skips_disabled_gate(tmp_path):
    _write_steps(tmp_path, _make_steps([100.0, 95.0, 90.0]))
    gates = {
        "drawdown": GateConfig(enabled=False, thresholds={"max_drawdown": 0.05}),
    }
    evaluator = StressGateEvaluator(gates)
    results = evaluator.evaluate(tmp_path)
    assert len(results) == 1
    assert results[0].status == "skipped"


def test_evaluator_unknown_gate_is_skipped(tmp_path):
    _write_steps(tmp_path, _make_steps([100.0]))
    gates = {"mystery_gate": GateConfig(enabled=True, thresholds={})}
    evaluator = StressGateEvaluator(gates)
    results = evaluator.evaluate(tmp_path)
    assert results[0].status == "skipped"
    assert "unknown gate" in (results[0].reason or "")


def test_evaluator_missing_steps_file(tmp_path):
    gates = {"drawdown": GateConfig(enabled=True, thresholds={"max_drawdown": 0.15})}
    evaluator = StressGateEvaluator(gates)
    # No steps.jsonl written — should not raise, passes with zero drawdown
    results = evaluator.evaluate(tmp_path)
    assert results[0].status == "pass"


def test_evaluator_all_gates(tmp_path):
    values = [100.0 + i * 0.5 for i in range(30)]  # steady growth, no drawdown
    _write_steps(tmp_path, _make_steps(values))
    gates = {
        "drawdown": GateConfig(enabled=True, thresholds={"max_drawdown": 0.15, "warning_threshold": 0.10}),
        "var_limit": GateConfig(enabled=True, thresholds={"var_95": 0.05, "var_99": 0.08}),
        "liquidity": GateConfig(enabled=False, thresholds={"min_daily_volume": 1_000_000}),
    }
    evaluator = StressGateEvaluator(gates)
    results = evaluator.evaluate(tmp_path)
    by_name = {r.gate_name: r for r in results}
    assert by_name["drawdown"].status == "pass"
    assert by_name["var_limit"].status == "pass"
    assert by_name["liquidity"].status == "skipped"


def test_load_steps_handles_malformed_lines(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    p = logs / "steps.jsonl"
    p.write_text('{"portfolio_value": 100.0}\nNOT_JSON\n{"portfolio_value": 101.0}\n', encoding="utf-8")
    steps = _load_steps(p)
    assert len(steps) == 2
    assert steps[0]["portfolio_value"] == 100.0
    assert steps[1]["portfolio_value"] == 101.0
