"""Integration tests for StressTestRunner.

Uses synthetic canonical data (same approach as test_execution_layer.py) to
exercise the full pipeline: data source → ShadowEngine → gate evaluation →
StressTestResult aggregation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research.experiments.registry import ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.stress.config import (
    GateConfig,
    HistoricalRegimeSpec,
    ScenarioSpec,
    StressTestConfig,
    SyntheticSpec,
)
from research.stress.runner import StressTestRunner
from research.stress.results import StressTestResult


# ---------------------------------------------------------------------------
# Shared fixture helpers (mirrored from test_execution_layer.py)
# ---------------------------------------------------------------------------

def _generate_rows(symbol: str, start: datetime, num_days: int, price_offset: int):
    rows = []
    for idx in range(num_days):
        ts = start + timedelta(days=idx)
        price = price_offset * 10 + idx
        rows.append({
            "symbol": symbol,
            "timestamp": ts.isoformat(),
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1_000 + idx,
        })
    return rows


def _write_canonical_year(data_root: Path, symbol: str, *, year: int, rows):
    base = data_root / "canonical" / "equity_ohlcv" / symbol / "daily"
    base.mkdir(parents=True, exist_ok=True)
    shard = base / f"{year}.parquet"
    shard.write_text(json.dumps(list(rows)), encoding="utf-8")


def _write_registry_artifacts(base: Path, spec: ExperimentSpec) -> None:
    for sub in ("spec", "evaluation", "logs", "promotion"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    (base / "spec" / "experiment_spec.json").write_text(spec.canonical_json, encoding="utf-8")
    metrics_payload = {
        "metadata": {"run_id": spec.experiment_id},
        "performance": {
            "total_return": 0.05,
            "cagr": None,
            "volatility_ann": 0.15,
            "sharpe": None,
            "max_drawdown": 0.03,
        },
        "trading": {"turnover_1d_mean": 0.0, "turnover_1d_median": 0.0, "avg_exposure": 1.0, "max_concentration": 0.5},
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": 0.0,
            "max_weight_violation_count": 0.0,
            "exposure_violation_count": 0.0,
            "turnover_violation_count": 0.0,
        },
    }
    (base / "evaluation" / "metrics.json").write_text(json.dumps(metrics_payload), encoding="utf-8")
    (base / "logs" / "run_summary.json").write_text(
        json.dumps({"recorded_at": "2024-01-01T00:00:00Z"}), encoding="utf-8"
    )
    (base / "promotion" / "qualification_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stress_env(tmp_path, monkeypatch):
    data_root = tmp_path / "quanto_data"
    monkeypatch.setenv("QUANTO_DATA_ROOT", str(data_root))

    symbols = ("AAA", "BBB")
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    num_days = 30

    for idx, symbol in enumerate(symbols):
        rows = _generate_rows(symbol, start, num_days, price_offset=1 + idx)
        _write_canonical_year(data_root, symbol, year=2020, rows=rows)

    end = start + timedelta(days=num_days - 1)
    spec = ExperimentSpec.from_mapping({
        "experiment_name": "stress_test_agent",
        "symbols": list(symbols),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {"fast_window": 1, "slow_window": 2, "policy_mode": "hard", "sigmoid_scale": 1.0},
        "cost_config": {"transaction_cost_bp": 0.0},
        "risk_config": {"max_weight": 1.0, "exposure_cap": 1.0, "min_cash": 0.0},
        "seed": 7,
    })

    registry_root = data_root / "experiments"
    registry = ExperimentRegistry(root=registry_root)
    registry_base = registry_root / spec.experiment_id
    _write_registry_artifacts(registry_base, spec)

    out_dir = tmp_path / "stress_runs"

    class Env:
        def __init__(self):
            self.spec = spec
            self.registry = registry
            self.data_root = data_root
            self.out_dir = out_dir

        def make_runner(self, config: StressTestConfig) -> StressTestRunner:
            return StressTestRunner(
                config,
                spec,
                out_dir=out_dir,
                registry=registry,
                data_root=data_root,
            )

    return Env()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_synthetic_shock_run_completes(stress_env):
    config = StressTestConfig(
        name="shock_test",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="shock", magnitude=-0.10, duration_days=5),
                name="10pct shock",
            )
        ],
        gates={
            "drawdown": GateConfig(enabled=True, thresholds={"max_drawdown": 0.99, "warning_threshold": 0.50}),
        },
        seeds=[42],
    )
    runner = stress_env.make_runner(config)
    result = runner.run()

    assert isinstance(result, StressTestResult)
    assert result.config_name == "shock_test"
    assert len(result.scenario_results) == 1
    run = result.scenario_results[0]
    assert run.status in {"pass", "warn", "fail"}
    assert run.error is None
    assert run.run_dir is not None and run.run_dir.exists()


def test_synthetic_vol_spike_run_completes(stress_env):
    config = StressTestConfig(
        name="vol_spike_test",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="volatility_spike", factor=2.0, duration_days=10),
                name="2x vol",
            )
        ],
        gates={
            "var_limit": GateConfig(enabled=True, thresholds={"var_95": 0.99, "var_99": 0.99}),
        },
        seeds=[1],
    )
    result = stress_env.make_runner(config).run()
    assert result.scenario_results[0].error is None


def test_multiple_seeds_produce_separate_runs(stress_env):
    config = StressTestConfig(
        name="multi_seed",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="shock", magnitude=-0.05, duration_days=3),
                name="small shock",
            )
        ],
        gates={},
        seeds=[1, 2, 3],
    )
    result = stress_env.make_runner(config).run()
    assert len(result.scenario_results) == 3
    run_dirs = {str(r.run_dir) for r in result.scenario_results}
    # Each seed should produce a distinct run directory
    assert len(run_dirs) == 3


def test_multiple_scenarios_all_executed(stress_env):
    config = StressTestConfig(
        name="multi_scenario",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="shock", magnitude=-0.10, duration_days=3),
                name="shock",
            ),
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="drift", magnitude=0.01, duration_days=5),
                name="drift",
            ),
        ],
        gates={},
        seeds=[42],
    )
    result = stress_env.make_runner(config).run()
    assert len(result.scenario_results) == 2
    names = {r.scenario_name for r in result.scenario_results}
    assert names == {"shock", "drift"}


def test_gate_fail_sets_overall_fail(stress_env):
    # Inject a shock mid-simulation so the agent is holding positions when prices drop.
    # Threshold of 0.00001 (0.001%) means any measurable drawdown will fail the gate.
    config = StressTestConfig(
        name="always_fail",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(
                    mutation="shock",
                    magnitude=-0.20,
                    duration_days=5,
                    inject_at="2020-01-15",  # shock after agent has built positions
                ),
                name="big shock",
            )
        ],
        gates={
            "drawdown": GateConfig(enabled=True, thresholds={"max_drawdown": 0.00001, "warning_threshold": 0.000001}),
        },
        seeds=[42],
    )
    result = stress_env.make_runner(config).run()
    assert result.overall_status == "fail"
    assert result.n_fail >= 1


def test_disabled_gates_all_skipped(stress_env):
    config = StressTestConfig(
        name="skipped_gates",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="shock", magnitude=-0.10, duration_days=3),
                name="shock",
            )
        ],
        gates={
            "drawdown": GateConfig(enabled=False, thresholds={"max_drawdown": 0.0}),
            "var_limit": GateConfig(enabled=False, thresholds={"var_95": 0.0}),
        },
        seeds=[42],
    )
    result = stress_env.make_runner(config).run()
    run = result.scenario_results[0]
    assert all(g.status == "skipped" for g in run.gate_results)
    assert result.overall_status == "pass"


def test_error_in_scenario_captured_not_raised(stress_env):
    # Inject a bad scenario type to force an error path without crashing the runner.
    bad_scenario = ScenarioSpec.__new__(ScenarioSpec)
    object.__setattr__(bad_scenario, "type", "nonexistent_type")
    object.__setattr__(bad_scenario, "historical_regime", None)
    object.__setattr__(bad_scenario, "synthetic", None)
    object.__setattr__(bad_scenario, "name", "bad_scenario")

    config = StressTestConfig(
        name="error_handling",
        scenarios=[bad_scenario],
        gates={},
        seeds=[42],
    )
    result = stress_env.make_runner(config).run()
    assert result.scenario_results[0].status == "error"
    assert result.scenario_results[0].error is not None
    assert result.overall_status == "fail"
    assert result.n_error == 1


def test_steps_jsonl_written_for_each_run(stress_env):
    config = StressTestConfig(
        name="artifact_check",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="shock", magnitude=-0.05, duration_days=3),
                name="shock",
            )
        ],
        gates={},
        seeds=[42],
    )
    result = stress_env.make_runner(config).run()
    run = result.scenario_results[0]
    assert run.run_dir is not None
    steps_path = run.run_dir / "logs" / "steps.jsonl"
    assert steps_path.exists()
    lines = [l for l in steps_path.read_text().splitlines() if l.strip()]
    assert len(lines) > 0
    first = json.loads(lines[0])
    assert "portfolio_value" in first


def test_result_to_dict_structure(stress_env):
    config = StressTestConfig(
        name="dict_check",
        scenarios=[
            ScenarioSpec(
                type="synthetic",
                synthetic=SyntheticSpec(mutation="shock", magnitude=-0.01, duration_days=2),
                name="tiny shock",
            )
        ],
        gates={"drawdown": GateConfig(enabled=True, thresholds={"max_drawdown": 0.99})},
        seeds=[42],
    )
    result = stress_env.make_runner(config).run()
    d = result.to_dict()
    assert "overall_status" in d
    assert "counts" in d
    assert d["counts"]["total"] == 1
    assert "runs" in d
    run_d = d["runs"][0]
    assert "scenario" in run_d
    assert "gates" in run_d
