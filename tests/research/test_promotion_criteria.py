from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from research.experiments.comparator import ComparisonResult, ComparisonSummary, MetricComparison
from research.experiments.registry import ExperimentRegistry
from research.promotion import criteria as criteria_mod
from research.promotion.criteria import QualificationCriteria, SweepRobustnessCriteria
from research.promotion.execution_criteria import ExecutionQualificationCriteria
from research.promotion.execution_metrics_locator import ExecutionMetricsLocatorResult
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
        "returns": [0.01, -0.02, 0.03],
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
             "turnover_1d_std": turnover,
             "turnover_1d_p95": turnover,
            "avg_exposure": 1.0,
            "max_concentration": 0.5,
            "hhi_mean": 0.3,
            "tx_cost_total": 5.0,
            "tx_cost_bps": 10.0,
            "avg_cash": 0.1,
            "cost_sensitivity_curve": {"0.5": 0.01, "1.0": 0.005, "1.5": 0.0},
        },
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": constraint_count,
            "max_weight_violation_count": 0.0,
            "exposure_violation_count": 0.0,
            "turnover_violation_count": 0.0,
        },
        "stability": {
            "turnover_std": turnover,
            "turnover_p95": turnover,
            "mode_churn_rate": 0.0,
            "mode_set_size": 1.0,
            "cost_curve_span": 0.0,
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


def test_sharpe_soft_gate_triggered(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, max_drawdown=0.05, turnover=0.05)
    _write_experiment(tmp_path, "candidate", sharpe=0.8, max_drawdown=0.04, turnover=0.03)
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5).evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert any(reason.startswith("soft_gate:sharpe_not_improved") for reason in evaluation.failed_soft)


def test_stability_regression_flagged(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "baseline", sharpe=1.0, max_drawdown=0.05, turnover=0.05)
    _write_experiment(tmp_path, "candidate", sharpe=1.1, max_drawdown=0.04, turnover=0.5)
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.6).evaluate(
        registry.get("candidate"),
        registry.get("baseline"),
        registry=registry,
    )
    assert any(reason.startswith("stability_regressed") for reason in evaluation.failed_soft)


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


def test_execution_delta_comparison_populates_summary():
    criteria = ExecutionQualificationCriteria()
    candidate_metrics = {
        "execution": {
            "summary": {
                "avg_slippage_bps": 6.0,
                "total_fees": 110.0,
                "turnover_realized": 0.55,
                "fill_rate": 0.995,
                "reject_rate": 0.001,
                "execution_halts": 0.0,
                "p95_slippage_bps": 8.0,
                "partial_fill_rate": 0.0,
            }
        }
    }
    baseline_metrics = {
        "execution": {
            "summary": {
                "avg_slippage_bps": 5.0,
                "total_fees": 100.0,
                "turnover_realized": 0.5,
                "fill_rate": 0.994,
                "reject_rate": 0.001,
                "execution_halts": 0.0,
                "p95_slippage_bps": 7.5,
                "partial_fill_rate": 0.0,
            }
        }
    }
    comparison = ComparisonResult(
        candidate_experiment_id="cand",
        baseline_experiment_id="base",
        metadata={},
        metrics=[],
        summary=ComparisonSummary(
            total_metrics=0,
            compared_metrics=0,
            num_improved=0,
            num_regressed=0,
            num_unchanged=0,
            num_missing=0,
        ),
        missing_metrics=[],
        regime_deltas={},
        execution_metrics={},
    )
    evaluation = criteria.evaluate(comparison, candidate_metrics, baseline_metrics)
    summary_block = evaluation.report["comparison"]["summary"]
    assert summary_block["turnover_realized"]["delta"] == pytest.approx(0.05)
    gate = next(entry for entry in evaluation.report["gates"] if entry["gate_id"] == "execution_turnover_drift")
    assert gate["status"] == "pass"
    assert gate["message"] != "baseline_missing"


def test_execution_delta_comparison_populates_summary():
    criteria = ExecutionQualificationCriteria()
    candidate_metrics = {
        "execution": {
            "summary": {
                "avg_slippage_bps": 6.0,
                "total_fees": 110.0,
                "turnover_realized": 0.55,
                "fill_rate": 0.995,
                "reject_rate": 0.001,
                "execution_halts": 0.0,
                "p95_slippage_bps": 8.0,
                "partial_fill_rate": 0.0,
            }
        }
    }
    baseline_metrics = {
        "execution": {
            "summary": {
                "avg_slippage_bps": 5.0,
                "total_fees": 100.0,
                "turnover_realized": 0.5,
                "fill_rate": 0.994,
                "reject_rate": 0.001,
                "execution_halts": 0.0,
                "p95_slippage_bps": 7.5,
                "partial_fill_rate": 0.0,
            }
        }
    }
    comparison = ComparisonResult(
        candidate_experiment_id="cand",
        baseline_experiment_id="base",
        metadata={},
        metrics=[],
        summary=ComparisonSummary(
            total_metrics=0,
            compared_metrics=0,
            num_improved=0,
            num_regressed=0,
            num_unchanged=0,
            num_missing=0,
        ),
        missing_metrics=[],
        regime_deltas={},
        execution_metrics={},
    )
    evaluation = criteria.evaluate(comparison, candidate_metrics, baseline_metrics)
    summary_block = evaluation.report["comparison"]["summary"]
    assert summary_block["avg_slippage_bps"]["delta"] == pytest.approx(1.0)
    gate = next(g for g in evaluation.report["gates"] if g["gate_id"] == "execution_avg_slippage_delta")
    assert gate["status"] == "pass"
    assert gate["message"] != "baseline_missing"


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


def test_execution_sharpe_correlation_handles_missing_slippage_delta():
    criteria = ExecutionQualificationCriteria()
    candidate_metrics = {
        "execution": {
            "summary": {
                "avg_slippage_bps": 6.0,
                "total_fees": 100.0,
                "turnover_realized": 0.5,
                "fill_rate": 0.995,
                "reject_rate": 0.001,
                "execution_halts": 0.0,
                "p95_slippage_bps": 7.5,
                "partial_fill_rate": 0.0,
            }
        }
    }
    baseline_metrics = {
        "execution": {
            "summary": {
                "avg_slippage_bps": 5.5,
                "total_fees": 100.0,
                "turnover_realized": 0.5,
                "fill_rate": 0.994,
                "reject_rate": 0.001,
                "execution_halts": 0.0,
                "p95_slippage_bps": 7.5,
                "partial_fill_rate": 0.0,
            }
        }
    }
    comparison = ComparisonResult(
        candidate_experiment_id="cand",
        baseline_experiment_id="base",
        metadata={},
        metrics=[
            MetricComparison(
                metric_id="performance.sharpe",
                category="performance",
                name="Sharpe",
                directionality="higher",
                candidate_value=0.8,
                baseline_value=1.0,
                delta=-0.2,
                delta_pct=None,
                status="regressed",
                reason=None,
            )
        ],
        summary=ComparisonSummary(
            total_metrics=1,
            compared_metrics=1,
            num_improved=0,
            num_regressed=1,
            num_unchanged=0,
            num_missing=0,
        ),
        missing_metrics=[],
        regime_deltas={},
        execution_metrics={
            "summary": {
                "avg_slippage_bps": {
                    "candidate": 6.0,
                    "baseline": 5.5,
                    "delta": None,
                    "delta_pct": None,
                }
            }
        },
    )
    evaluation = criteria.evaluate(comparison, candidate_metrics, baseline_metrics)
    gate = next(entry for entry in evaluation.report["gates"] if entry["gate_id"] == "execution_sharpe_correlation")
    assert gate["status"] == "pass"


def test_phase1_delta_gates_skipped_with_warning(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
    _write_experiment(tmp_path, "phase1", sharpe=1.0, max_drawdown=0.05, turnover=0.05)
    record = registry.get("phase1")
    evaluation = QualificationCriteria(max_drawdown=0.5, max_turnover=0.5).evaluate(
        record,
        record,
        registry=registry,
    )
    assert evaluation.passed is True
    assert any(
        entry == "Baseline equals candidate; delta-based gates skipped (Phase 1 mode)."
        for entry in evaluation.failed_soft
    )
    exec_report = evaluation.execution_report or {}
    gate_messages = [gate.get("message") for gate in exec_report.get("gates", []) if gate.get("severity") == "soft"]
    assert "phase1_delta_skip" in gate_messages
    resolution = evaluation.execution_resolution or {}
    assert resolution["candidate"]["found"] is True
    assert resolution["baseline"]["found"] is True


def test_execution_metrics_missing_records_attempts(tmp_path: Path):
    registry = ExperimentRegistry(root=tmp_path)
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
        registry.get("candidate"),
        registry=registry,
    )
    assert "execution_metrics_missing" in evaluation.failed_hard
    resolution = evaluation.execution_resolution or {}
    candidate_entry = resolution.get("candidate") or {}
    assert candidate_entry.get("found") is False
    assert candidate_entry.get("attempted") is not None


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
