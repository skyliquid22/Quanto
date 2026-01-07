from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.experiments.ablation import SweepExperiment, SweepResult
from research.experiments.aggregate import aggregate_sweep
from research.experiments.registry import ExperimentRegistry
from research.experiments.spec import ExperimentSpec
from research.experiments.sweep import SweepSpec, expand_sweep_entries


def _base_spec() -> ExperimentSpec:
    payload = {
        "experiment_name": "aggregation_test",
        "symbols": ["AAPL"],
        "start_date": "2023-01-01",
        "end_date": "2023-02-01",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "sma",
        "policy_params": {"fast_window": 2, "slow_window": 4},
        "cost_config": {"transaction_cost_bp": 1.0},
        "seed": 0,
    }
    return ExperimentSpec.from_mapping(payload)


def _write_registry_entry(root: Path, exp: SweepExperiment, metrics_payload: dict, gate_payload: dict | None) -> None:
    base = root / exp.experiment_id
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    (base / "spec" / "experiment_spec.json").write_text(
        json.dumps(exp.spec.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (base / "evaluation" / "metrics.json").write_text(
        json.dumps(metrics_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    if gate_payload:
        (base / "comparison").mkdir(parents=True, exist_ok=True)
        (base / "comparison" / "gate_report.json").write_text(
            json.dumps(gate_payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )


def _metrics_bundle(sharpe: float, total_return: float, drawdown: float, turnover: float) -> dict:
    return {
        "metadata": {
            "symbols": ["AAPL"],
            "feature_set": "sma_v1",
            "start_date": "2023-01-01",
            "end_date": "2023-02-01",
            "interval": "daily",
        },
        "performance": {
            "total_return": total_return,
            "cagr": None,
            "volatility_ann": 0.2,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "calmar": None,
        },
        "trading": {
            "turnover_1d_mean": turnover,
            "turnover_1d_median": turnover,
            "avg_exposure": 1.0,
            "max_concentration": 0.5,
            "hhi_mean": 0.4,
            "tx_cost_total": 2.0,
            "tx_cost_bps": 10.0,
        },
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
        },
    }


def _gate_payload(status: str, warning_count: int) -> dict:
    return {
        "overall_status": status,
        "hard_failures": 1 if status == "fail" else 0,
        "soft_warnings": warning_count,
        "evaluations": [
            {"gate_id": "sharpe_guard", "status": status, "metric_id": "performance.sharpe"},
        ],
    }


def test_aggregate_sweep(tmp_path: Path):
    base_spec = _base_spec()
    sweep_payload = {
        "sweep_name": "ablation_demo",
        "base_experiment_spec": base_spec.to_dict(),
        "sweep_dimensions": {
            "feature_set": ["sma_v1", "sma_v2"],
            "seed": [0, 1],
        },
    }
    sweep_spec = SweepSpec.from_mapping(sweep_payload)
    expansions = expand_sweep_entries(sweep_spec)
    experiments = [
        SweepExperiment(
            experiment_id=entry.spec.experiment_id,
            spec=entry.spec,
            dimensions=entry.dimension_values,
            status="completed",
        )
        for entry in expansions[:3]
    ]
    result = SweepResult(sweep_spec=sweep_spec, experiments=tuple(experiments))

    registry_root = tmp_path / "experiments"
    registry = ExperimentRegistry(root=registry_root)
    for idx, experiment in enumerate(experiments):
        sharpe = [1.0, 0.4, 1.6][idx]
        total_return = [0.12, 0.05, 0.2][idx]
        drawdown = [0.08, 0.1, 0.06][idx]
        turnover = [0.4, 0.5, 0.35][idx]
        gate_payload = None
        if idx == 0:
            gate_payload = _gate_payload("pass", 0)
        elif idx == 1:
            gate_payload = _gate_payload("fail", 1)
        _write_registry_entry(
            registry_root,
            experiment,
            _metrics_bundle(sharpe, total_return, drawdown, turnover),
            gate_payload,
        )

    aggregation = aggregate_sweep(result, registry=registry)
    metrics_summary = aggregation.metrics_summary
    regression_summary = aggregation.regression_summary

    sharpe_stats = metrics_summary["metrics"]["performance.sharpe"]
    assert sharpe_stats["samples"] == 3
    assert sharpe_stats["mean"] == pytest.approx(1.0, rel=1e-9)
    assert sharpe_stats["std"] == pytest.approx(0.48989794855663565, rel=1e-9)

    turnover_stats = metrics_summary["metrics"]["trading.turnover_1d_mean"]
    assert turnover_stats["median"] == 0.4

    configuration_counts = metrics_summary["configuration_counts"]
    assert configuration_counts[0]["sample_count"] == 1

    assert regression_summary["passes"] == 1
    assert regression_summary["fails"] == 1
    assert regression_summary["reports_evaluated"] == 2
    assert experiments[2].experiment_id in regression_summary["missing_reports"]
