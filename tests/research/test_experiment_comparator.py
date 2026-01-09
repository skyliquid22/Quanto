from __future__ import annotations

import json
from pathlib import Path

from research.experiments.comparator import compare_experiments
from research.experiments.registry import ExperimentRegistry


def _write_experiment(root: Path, experiment_id: str, sharpe: float, drawdown: float, tx_cost_total: float | None):
    base = root / experiment_id
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    spec_path = base / "spec" / "experiment_spec.json"
    spec_path.write_text(json.dumps({"experiment_name": experiment_id}), encoding="utf-8")
    log_path = base / "logs" / "run_summary.json"
    log_path.write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00+00:00"}), encoding="utf-8")
    metrics = {
        "metadata": {
            "symbols": ["AAA"],
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "interval": "daily",
            "feature_set": "test_features",
            "policy_id": "equal_weight",
            "run_id": experiment_id,
        },
        "performance": {
            "total_return": 0.12 if experiment_id == "candidate" else 0.1,
            "cagr": 0.11 if experiment_id == "candidate" else 0.1,
            "volatility_ann": 0.3,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "calmar": 1.1,
        },
        "trading": {
            "turnover_1d_mean": 0.05,
            "turnover_1d_median": 0.04,
            "turnover_1d_std": 0.02,
            "turnover_1d_p95": 0.06,
            "avg_exposure": 1.0,
            "max_concentration": 0.6,
            "hhi_mean": 0.45,
            "tx_cost_total": tx_cost_total,
            "tx_cost_bps": 15.0,
            "avg_cash": 0.1,
        },
        "stability": {
            "turnover_std": 0.02,
            "mode_churn_rate": 0.0,
            "mode_set_size": 1.0,
            "cost_curve_span": 0.0,
        },
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": 0.0,
            "max_weight_violation_count": 0.0,
            "exposure_violation_count": 0.0,
            "turnover_violation_count": 0.0,
        },
        "execution": {
            "summary": {
                "fill_rate": 0.99 if experiment_id == "candidate" else 0.995,
                "reject_rate": 0.01 if experiment_id == "candidate" else 0.005,
                "avg_slippage_bps": 4.0 if experiment_id == "candidate" else 2.0,
                "p95_slippage_bps": 6.0 if experiment_id == "candidate" else 3.0,
                "total_fees": 50.0 if experiment_id == "candidate" else 40.0,
                "turnover_realized": 0.6 if experiment_id == "candidate" else 0.5,
                "execution_halts": 0.0,
                "halt_reasons": [],
                "order_latency_ms": {},
                "partial_fill_rate": 0.0,
            },
            "regime": {
                "high_vol": {
                    "avg_slippage_bps": 5.0 if experiment_id == "candidate" else 4.0,
                    "p95_slippage_bps": 7.0 if experiment_id == "candidate" else 5.0,
                    "reject_rate": 0.01,
                    "fill_rate": 0.99,
                },
                "mid_vol": {"avg_slippage_bps": 2.0, "p95_slippage_bps": 3.0, "reject_rate": 0.01, "fill_rate": 0.99},
                "low_vol": {"avg_slippage_bps": 1.0, "p95_slippage_bps": 2.0, "reject_rate": 0.01, "fill_rate": 0.99},
            },
        },
        "returns": [0.0, 0.01, 0.0],
    }
    metrics_path = base / "evaluation" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True, indent=2), encoding="utf-8")


def test_compare_experiments_deltas(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, drawdown=0.07, tx_cost_total=10.0)
    _write_experiment(tmp_path, "candidate", sharpe=1.15, drawdown=0.08, tx_cost_total=None)

    result = compare_experiments("candidate", "baseline", registry=registry)

    sharpe_entry = next(item for item in result.metrics if item.metric_id == "performance.sharpe")
    assert sharpe_entry.delta == 0.15
    assert sharpe_entry.delta_pct == 15.0
    assert sharpe_entry.status == "improved"

    max_dd_entry = next(item for item in result.metrics if item.metric_id == "performance.max_drawdown")
    assert max_dd_entry.delta == 0.01
    assert max_dd_entry.status == "regressed"  # higher drawdown is worse

    tx_cost_entry = next(item for item in result.metrics if item.metric_id == "trading.tx_cost_total")
    assert tx_cost_entry.reason == "candidate_missing"

    assert result.summary.num_improved >= 1
    assert result.summary.num_regressed >= 1
    assert result.summary.num_missing == 1

    ordered_ids = [entry.metric_id for entry in result.metrics]
    assert ordered_ids[0] == "performance.total_return"
    assert ordered_ids[-1] == "execution.summary.partial_fill_rate"

    exec_summary = result.execution_metrics["summary"]
    assert exec_summary["fill_rate"]["candidate"] == 0.99
    assert exec_summary["fill_rate"]["baseline"] == 0.995
    assert exec_summary["fill_rate"]["delta"] == -0.005
    high_vol_exec = result.execution_metrics["regime"]["high_vol"]["avg_slippage_bps"]
    assert high_vol_exec["candidate"] == 5.0
    assert high_vol_exec["baseline"] == 4.0
    assert high_vol_exec["delta"] == 1.0
