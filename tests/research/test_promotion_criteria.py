from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from research.experiments.registry import ExperimentRegistry
from research.promotion.criteria import QualificationCriteria, SweepRobustnessCriteria
from research.promotion.regime_criteria import RegimeQualificationCriteria


def _write_experiment(
    root: Path,
    experiment_id: str,
    *,
    sharpe: float,
    max_drawdown: float,
    turnover: float,
    constraint_count: float = 0.0,
    execution_summary: Mapping[str, float | int] | None = None,
    execution_regime: Mapping[str, Mapping[str, float]] | None = None,
    performance_by_regime: Mapping[str, Mapping[str, float | None]] | None = None,
    include_execution: bool = True,
) -> None:
    base = root / experiment_id
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    spec_payload = {"experiment_name": experiment_id}
    (base / "spec" / "experiment_spec.json").write_text(json.dumps(spec_payload), encoding="utf-8")
    metrics_payload = {
        "metadata": {"run_id": experiment_id},
        "performance": {
            "total_return": 0.12,
            "cagr": None,
            "volatility_ann": 0.2,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": None,
        },
        "trading": {
            "turnover_1d_mean": turnover,
            "turnover_1d_median": turnover,
            "avg_exposure": 1.0,
            "max_concentration": 0.5,
            "hhi_mean": 0.3,
            "tx_cost_total": 5.0,
            "tx_cost_bps": 10.0,
            "avg_cash": 0.1,
        },
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": constraint_count,
            "max_weight_violation_count": 0.0,
            "exposure_violation_count": 0.0,
            "turnover_violation_count": 0.0,
        },
    }
    metrics_payload["performance_by_regime"] = performance_by_regime or _regime_performance_payload(max_drawdown)
    if include_execution:
        metrics_payload["execution"] = _execution_payload(execution_summary, execution_regime)
    (base / "evaluation" / "metrics.json").write_text(
        json.dumps(metrics_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (base / "logs" / "run_summary.json").write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00Z"}), encoding="utf-8")


def _execution_payload(
    summary_override: Mapping[str, float | int] | None,
    regime_override: Mapping[str, Mapping[str, float]] | None,
) -> Mapping[str, Any]:
    summary = {
        "fill_rate": 0.995,
        "reject_rate": 0.001,
        "avg_slippage_bps": 1.0,
        "p95_slippage_bps": 5.0,
        "total_fees": 100.0,
        "turnover_realized": 0.4,
        "execution_halts": 0.0,
        "halt_reasons": [],
        "order_latency_ms": {},
        "partial_fill_rate": 0.0,
    }
    if summary_override:
        summary.update(summary_override)
    regime_defaults = {
        bucket: {
            "avg_slippage_bps": 1.0,
            "p95_slippage_bps": 5.0,
            "reject_rate": 0.001,
            "fill_rate": 0.995,
        }
        for bucket in ("high_vol", "mid_vol", "low_vol")
    }
    if regime_override:
        for bucket, payload in regime_override.items():
            if bucket in regime_defaults:
                regime_defaults[bucket].update(payload)
    return {"summary": summary, "regime": regime_defaults}


def _regime_performance_payload(max_drawdown: float) -> Mapping[str, Mapping[str, float | None]]:
    return {
        "high_vol": {
            "total_return": 0.05,
            "max_drawdown": max_drawdown,
            "volatility_ann": 0.1,
            "avg_exposure": 0.6,
            "avg_turnover": 0.3,
            "sharpe": 1.1,
        }
    }


def test_qualification_criteria_pass(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, max_drawdown=0.1, turnover=0.05)
    _write_experiment(tmp_path, "candidate", sharpe=1.05, max_drawdown=0.09, turnover=0.04)
    baseline = registry.get("baseline")
    candidate = registry.get("candidate")

    criteria = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5)
    evaluation = criteria.evaluate(candidate, baseline, registry=registry)

    assert evaluation.passed is True
    assert evaluation.failed_hard == []
    assert evaluation.failed_soft == []
    assert evaluation.gate_summary["overall_status"] == "pass"
    assert evaluation.metrics_snapshot["performance"]["sharpe"] == 1.05


def test_qualification_detects_sweep_and_constraints(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path / "experiments")
    registry.root.mkdir()
    _write_experiment(registry.root, "baseline", sharpe=1.0, max_drawdown=0.08, turnover=0.05)
    _write_experiment(
        registry.root,
        "candidate",
        sharpe=1.02,
        max_drawdown=0.09,
        turnover=0.04,
        constraint_count=1.0,
    )
    sweep_root = tmp_path / "sweeps"
    sweep_dir = sweep_root / "demo_sweep"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate_metrics.json").write_text(json.dumps({"completed": 1}, indent=2), encoding="utf-8")
    (sweep_dir / "regression_summary.json").write_text(json.dumps({"fails": 2}, indent=2), encoding="utf-8")

    criteria = QualificationCriteria(
        max_drawdown=0.5,
        max_turnover=0.5,
        sweep_root=sweep_root,
        sweep_robustness=SweepRobustnessCriteria(min_completed=2, severity="soft"),
    )
    evaluation = criteria.evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
        sweep_name="demo_sweep",
    )

    assert evaluation.passed is False
    assert "constraint_violation:constraint_violations_count" in "|".join(evaluation.failed_hard)
    assert any(reason.startswith("sweep_completed_below_threshold") for reason in evaluation.failed_soft)
    assert evaluation.sweep_summary is not None


def test_execution_metrics_missing_fails(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, max_drawdown=0.1, turnover=0.05)
    _write_experiment(
        tmp_path,
        "candidate",
        sharpe=1.0,
        max_drawdown=0.1,
        turnover=0.05,
        include_execution=False,
    )
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5).evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert "execution_metrics_missing" in evaluation.failed_hard


def test_execution_metrics_malformed(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, max_drawdown=0.05, turnover=0.05)
    _write_experiment(
        tmp_path,
        "candidate",
        sharpe=1.0,
        max_drawdown=0.05,
        turnover=0.05,
        execution_summary={"fill_rate": "invalid"},
    )
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5).evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert any(reason.startswith("missing_execution_metric:execution.summary.fill_rate") for reason in evaluation.failed_hard)


def test_execution_reject_rate_gate(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, max_drawdown=0.05, turnover=0.05)
    _write_experiment(
        tmp_path,
        "candidate",
        sharpe=1.0,
        max_drawdown=0.05,
        turnover=0.05,
        execution_summary={"reject_rate": 0.05},
    )
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5).evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert "execution_reject_rate_exceeded" in evaluation.failed_hard


def test_execution_slippage_soft_gate(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, max_drawdown=0.05, turnover=0.05)
    _write_experiment(
        tmp_path,
        "candidate",
        sharpe=1.0,
        max_drawdown=0.05,
        turnover=0.05,
        execution_summary={"avg_slippage_bps": 7.0},
    )
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5).evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert any("execution.summary.avg_slippage_bps_regressed" in reason for reason in evaluation.failed_soft)


def test_execution_sharpe_correlation_warning(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.1, max_drawdown=0.05, turnover=0.05)
    _write_experiment(
        tmp_path,
        "candidate",
        sharpe=0.8,
        max_drawdown=0.05,
        turnover=0.05,
        execution_summary={"avg_slippage_bps": 6.0},
    )
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5).evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert "execution_sharpe_correlated_with_slippage" in evaluation.failed_soft


def test_execution_high_vol_drawdown_gate(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(
        tmp_path,
        "baseline",
        sharpe=1.0,
        max_drawdown=0.07,
        turnover=0.05,
        performance_by_regime=_regime_performance_payload(0.07),
    )
    candidate_regime = _regime_performance_payload(0.12)
    _write_experiment(
        tmp_path,
        "candidate",
        sharpe=1.0,
        max_drawdown=0.12,
        turnover=0.05,
        performance_by_regime=candidate_regime,
    )
    criteria = QualificationCriteria(
        max_drawdown=0.5,
        max_turnover=0.5,
        regime_qualification=RegimeQualificationCriteria(execution_drawdown_regression_pct=2.0),
    )
    evaluation = criteria.evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert "execution_high_vol_drawdown_regressed" in evaluation.failed_hard
