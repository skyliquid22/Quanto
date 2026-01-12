from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Mapping, Tuple

import scripts.run_v1_release_candidate as runner
from research.experiments.runner import ExperimentResult
from research.promotion.qualify import QualificationResult
from research.promotion.report import QualificationReport


def make_args(spec_path: Path, data_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        spec=str(spec_path),
        baseline="baseline_case",
        data_root=str(data_root),
        registry_root=str(data_root / "experiments"),
        promotion_root=str(data_root / "promotions"),
        release_root=str(data_root / "v1_release"),
        gate_config=None,
        promotion_tier="candidate",
        promotion_reason="integration-test",
        shadow_start_date=None,
        shadow_end_date=None,
        shadow_max_steps=None,
        force_experiment=True,
    )


def write_spec(path: Path) -> Path:
    payload = {
        "experiment_name": "ppo_release_ci",
        "symbols": ["AAA"],
        "start_date": "2023-01-01",
        "end_date": "2023-01-05",
        "interval": "daily",
        "feature_set": "sma_v1",
        "policy": "ppo",
        "policy_params": {
            "fast_window": 2,
            "slow_window": 3,
            "timesteps": 10,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "reward_version": "reward_v1",
        },
        "cost_config": {"transaction_cost_bp": 1.0},
        "risk_config": {"max_weight": 1.0, "exposure_cap": 1.0, "min_cash": 0.0},
        "seed": 7,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def patch_run_experiment(monkeypatch):
    def fake_run_experiment(spec, registry, force=False, data_root=None):
        paths = registry.prepare(spec.experiment_id, force=True)
        registry.write_spec(spec, paths)
        metrics = {
            "metadata": {"run_id": spec.experiment_id},
            "performance": {"total_return": 0.01},
            "trading": {"turnover_1d_mean": 0.05},
        }
        metrics_path = paths.evaluation_dir / "metrics.json"
        paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, sort_keys=True, indent=2), encoding="utf-8")
        paths.runs_dir.mkdir(parents=True, exist_ok=True)
        rollout_artifact = paths.runs_dir / "rollout.json"
        rollout_artifact.write_text(json.dumps({"weights": [0.0, 1.0]}), encoding="utf-8")
        registry.write_run_summary(
            paths,
            {
                "experiment_id": spec.experiment_id,
                "policy": spec.policy,
                "metrics_path": str(metrics_path),
            },
        )
        return ExperimentResult(
            experiment_id=spec.experiment_id,
            registry_paths=paths,
            metrics_path=metrics_path,
            evaluation_payload=metrics,
            evaluation_summary={},
            training_artifacts={},
            rollout_artifact=rollout_artifact,
        )

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)


def patch_run_qualification(monkeypatch, *, passed: bool):
    def fake_run_qualification(experiment_id, baseline_identifier, registry, gate_rules=None, sweep_name=None, criteria=None):
        record = registry.get(experiment_id)
        record.promotion_dir.mkdir(parents=True, exist_ok=True)
        report = QualificationReport(
            experiment_id=experiment_id,
            baseline_experiment_id=str(baseline_identifier),
            passed=passed,
            failed_hard=[] if passed else ["regression_gates_failed"],
            failed_soft=[],
            metrics_snapshot={"performance": {"sharpe": 1.0}},
            gate_summary={"status": "pass" if passed else "fail"},
        )
        report_path = record.promotion_dir / "qualification_report.json"
        report_path.write_text(json.dumps(report.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
        return QualificationResult(report=report, report_path=report_path)

    monkeypatch.setattr(runner, "run_qualification", fake_run_qualification)


def patch_shadow(monkeypatch) -> List[Mapping[str, Any]]:
    calls: List[Mapping[str, Any]] = []

    def fake_shadow(spec, registry, data_root, shadow_dir, run_id, promotion_root, start_date, end_date, max_steps):
        shadow_dir.mkdir(parents=True, exist_ok=True)
        state_path = shadow_dir / "state.json"
        state_path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        log_path = shadow_dir / "logs.jsonl"
        log_path.write_text("{}", encoding="utf-8")
        summary = {
            "experiment_id": spec.experiment_id,
            "run_id": run_id,
            "steps_executed": 3,
            "completed": True,
            "state_path": str(state_path),
            "log_path": str(log_path),
        }
        summary_path = shadow_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
        calls.append(summary)
        return summary

    monkeypatch.setattr(runner, "_run_shadow_replay", fake_shadow)
    return calls


def snapshot_dir(base: Path) -> List[Tuple[str, bytes]]:
    entries: List[Tuple[str, bytes]] = []
    if not base.exists():
        return entries
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        entries.append((rel, path.read_bytes()))
    return entries


__all__ = [
    "make_args",
    "patch_run_experiment",
    "patch_run_qualification",
    "patch_shadow",
    "snapshot_dir",
    "write_spec",
]
