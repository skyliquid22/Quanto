from __future__ import annotations

import json
from pathlib import Path

from research.experiments.registry import ExperimentRegistry
from research.experiments.regression import GateType, RegressionGateRule
from research.promotion.criteria import QualificationCriteria
from research.promotion.regime_criteria import RegimeQualificationCriteria
from research.promotion.qualify import run_qualification


def _write_experiment(
    registry_root: Path,
    experiment_id: str,
    *,
    sharpe: float,
    high_vol_drawdown: float,
    high_vol_exposure: float,
    high_vol_turnover: float,
    hierarchy_enabled: bool,
) -> None:
    base = registry_root / experiment_id
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    spec_payload = {
        "experiment_name": experiment_id,
        "symbols": ["AAPL"],
        "start_date": "2023-01-01",
        "end_date": "2023-01-10",
        "interval": "daily",
        "feature_set": "demo",
        "policy": "equal_weight",
        "policy_params": {},
        "cost_config": {"transaction_cost_bp": 1},
        "risk_config": {},
        "seed": 7,
        "hierarchy_enabled": hierarchy_enabled,
        "controller_config": {"type": "linear"},
        "allocator_by_mode": {
            "risk_on": {"type": "equal_weight"},
            "neutral": {"type": "equal_weight"},
            "defensive": {"type": "equal_weight"},
        },
    }
    (base / "spec" / "experiment_spec.json").write_text(json.dumps(spec_payload), encoding="utf-8")
    metrics_payload = _metrics_payload(
        sharpe=sharpe,
        high_vol_drawdown=high_vol_drawdown,
        high_vol_exposure=high_vol_exposure,
        high_vol_turnover=high_vol_turnover,
    )
    (base / "evaluation" / "metrics.json").write_text(
        json.dumps(metrics_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (base / "logs" / "run_summary.json").write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00Z"}), encoding="utf-8")


def _metrics_payload(
    *,
    sharpe: float,
    high_vol_drawdown: float,
    high_vol_exposure: float,
    high_vol_turnover: float,
) -> dict:
    return {
        "metadata": {"run_id": "test"},
        "returns": [0.01, -0.02, 0.03],
        "performance": {
            "total_return": 0.15,
            "cagr": None,
            "volatility_ann": 0.3,
            "sharpe": sharpe,
            "max_drawdown": 0.4,
            "calmar": None,
        },
        "trading": {
            "turnover_1d_mean": 0.15,
            "turnover_1d_median": 0.14,
            "turnover_1d_std": 0.12,
            "turnover_1d_p95": 0.2,
            "avg_exposure": 0.9,
            "max_concentration": 0.5,
            "hhi_mean": 0.4,
            "tx_cost_total": 1.0,
            "tx_cost_bps": 5.0,
            "avg_cash": 0.1,
            "mode_counts": {"risk_on": 40, "neutral": 40, "defensive": 20},
            "mode_transitions": {"risk_on->neutral": 4, "neutral->defensive": 4},
            "cost_sensitivity_curve": {"0.5": 0.01, "1.0": 0.0, "1.5": -0.01},
        },
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": 0.0,
            "max_weight_violation_count": 0.0,
            "exposure_violation_count": 0.0,
            "turnover_violation_count": 0.0,
        },
        "regime_slicing": {"signal": "market_vol_20d", "quantiles": {"q33": 0.2, "q66": 0.7}},
        "performance_by_regime": {
            "high_vol": {
                "total_return": 0.05,
                "max_drawdown": high_vol_drawdown,
                "volatility_ann": 0.35,
                "avg_exposure": high_vol_exposure,
                "avg_turnover": high_vol_turnover,
                "sharpe": 0.8,
            },
            "mid_vol": {
                "total_return": 0.04,
                "max_drawdown": 0.2,
                "volatility_ann": 0.25,
                "avg_exposure": 0.5,
                "avg_turnover": 0.08,
                "sharpe": 1.0,
            },
            "low_vol": {
                "total_return": 0.03,
                "max_drawdown": 0.15,
                "volatility_ann": 0.2,
                "avg_exposure": 0.6,
                "avg_turnover": 0.07,
                "sharpe": 1.2,
            },
        },
        "stability": {
            "turnover_std": 0.12,
            "turnover_p95": 0.2,
            "mode_churn_rate": 0.1,
            "mode_set_size": 3.0,
            "cost_curve_span": 0.02,
        },
        "execution": {
            "summary": {
                "fill_rate": 0.995,
                "reject_rate": 0.002,
                "avg_slippage_bps": 2.0,
                "p95_slippage_bps": 4.0,
                "total_fees": 20.0,
                "turnover_realized": 0.3,
                "execution_halts": 0.0,
                "halt_reasons": [],
                "order_latency_ms": {},
                "partial_fill_rate": 0.0,
            },
            "regime": {
                "high_vol": {"avg_slippage_bps": 2.5, "p95_slippage_bps": 5.0, "reject_rate": 0.002, "fill_rate": 0.995},
                "mid_vol": {"avg_slippage_bps": 1.5, "p95_slippage_bps": 3.0, "reject_rate": 0.002, "fill_rate": 0.995},
                "low_vol": {"avg_slippage_bps": 1.0, "p95_slippage_bps": 2.0, "reject_rate": 0.002, "fill_rate": 0.995},
            },
        },
    }


def test_regime_qualification_with_warnings(tmp_path: Path):
    registry_root = tmp_path / "experiments"
    registry_root.mkdir()
    baseline_id = "baseline_regime"
    candidate_id = "candidate_regime"
    _write_experiment(
        registry_root,
        baseline_id,
        sharpe=1.0,
        high_vol_drawdown=0.3,
        high_vol_exposure=0.45,
        high_vol_turnover=0.12,
        hierarchy_enabled=True,
    )
    _write_experiment(
        registry_root,
        candidate_id,
        sharpe=0.8,
        high_vol_drawdown=0.25,
        high_vol_exposure=0.35,
        high_vol_turnover=0.13,
        hierarchy_enabled=True,
    )
    registry = ExperimentRegistry(root=registry_root)
    criteria = QualificationCriteria(regime_qualification=RegimeQualificationCriteria())
    relaxed_gates = [
        RegressionGateRule(
            gate_id="sharpe_guard_relaxed",
            metric_id="performance.sharpe",
            gate_type=GateType.HARD,
            max_degradation_pct=25.0,
        ),
        RegressionGateRule(
            gate_id="nan_guard",
            metric_id="safety.nan_inf_violations",
            gate_type=GateType.HARD,
            max_value=0.0,
        ),
    ]
    result = run_qualification(
        candidate_id,
        baseline_id,
        registry=registry,
        criteria=criteria,
        gate_rules=relaxed_gates,
    )
    assert result.report.passed is True
    assert any("regime_sharpe_degradation" in entry for entry in result.report.failed_soft)
    assert not result.report.failed_hard
    report_payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    high_checks = report_payload["regime_qualification"]["high_vol"]["checks"]
    assert high_checks["drawdown_protection"]["status"] == "pass"
    assert high_checks["global_sharpe_guard"]["status"] == "warn"

    # Update candidate metrics to violate high-vol drawdown protection.
    _write_experiment(
        registry_root,
        candidate_id,
        sharpe=0.9,
        high_vol_drawdown=0.45,
        high_vol_exposure=0.35,
        high_vol_turnover=0.13,
        hierarchy_enabled=True,
    )
    result_fail = run_qualification(
        candidate_id,
        baseline_id,
        registry=registry,
        criteria=criteria,
        gate_rules=relaxed_gates,
    )
    assert result_fail.report.passed is False
    assert "regime_high_vol_drawdown_regressed" in result_fail.report.failed_hard
