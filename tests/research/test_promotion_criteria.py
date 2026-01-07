from __future__ import annotations

import json
from pathlib import Path

from research.experiments.registry import ExperimentRegistry
from research.promotion.criteria import QualificationCriteria, SweepRobustnessCriteria


def _write_experiment(
    root: Path,
    experiment_id: str,
    *,
    sharpe: float,
    max_drawdown: float,
    turnover: float,
    constraint_count: float = 0.0,
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
    (base / "evaluation" / "metrics.json").write_text(
        json.dumps(metrics_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (base / "logs" / "run_summary.json").write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00Z"}), encoding="utf-8")


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
