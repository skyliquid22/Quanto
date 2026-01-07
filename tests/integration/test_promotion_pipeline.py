from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_experiment(
    registry_root: Path,
    experiment_id: str,
    *,
    sharpe: float,
    max_drawdown: float,
    turnover: float,
) -> None:
    base = registry_root / experiment_id
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    spec_payload = {"experiment_name": experiment_id}
    (base / "spec" / "experiment_spec.json").write_text(json.dumps(spec_payload), encoding="utf-8")
    metrics_payload = {
        "metadata": {"run_id": experiment_id},
        "performance": {
            "total_return": 0.15,
            "cagr": None,
            "volatility_ann": 0.25,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": None,
        },
        "trading": {
            "turnover_1d_mean": turnover,
            "turnover_1d_median": turnover,
            "avg_exposure": 1.0,
            "max_concentration": 0.4,
            "hhi_mean": 0.3,
            "tx_cost_total": 2.0,
            "tx_cost_bps": 11.0,
            "avg_cash": 0.1,
        },
        "safety": {
            "nan_inf_violations": 0.0,
            "action_bounds_violations": 0.0,
            "constraint_violations_count": 0.0,
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


def test_promotion_pipeline(tmp_path: Path):
    registry_root = tmp_path / "experiments"
    registry_root.mkdir()
    baseline_id = "baseline_case"
    candidate_id = "candidate_case"
    _write_experiment(registry_root, baseline_id, sharpe=1.0, max_drawdown=0.08, turnover=0.05)
    _write_experiment(registry_root, candidate_id, sharpe=1.05, max_drawdown=0.07, turnover=0.04)

    qualify_cmd = [
        sys.executable,
        "scripts/qualify_experiment.py",
        "--experiment-id",
        candidate_id,
        "--baseline",
        baseline_id,
        "--registry-root",
        str(registry_root),
    ]
    qualify_run = subprocess.run(qualify_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert qualify_run.returncode == 0, qualify_run.stdout + qualify_run.stderr
    report_path = registry_root / candidate_id / "promotion" / "qualification_report.json"
    assert report_path.exists()
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["passed"] is True
    assert report_payload["failed_hard"] == []

    promotion_root = tmp_path / "promotions"
    promote_cmd = [
        sys.executable,
        "scripts/promote_experiment.py",
        "--experiment-id",
        candidate_id,
        "--tier",
        "candidate",
        "--reason",
        "integration test approval",
        "--registry-root",
        str(registry_root),
        "--promotion-root",
        str(promotion_root),
    ]
    promote_run = subprocess.run(promote_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert promote_run.returncode == 0, promote_run.stdout + promote_run.stderr
    promotion_path = promotion_root / "candidate" / f"{candidate_id}.json"
    assert promotion_path.exists()
    promotion_payload = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert promotion_payload["experiment_id"] == candidate_id
    assert promotion_payload["tier"] == "candidate"
    assert promotion_payload["promotion_reason"] == "integration test approval"

    # Idempotent writes should succeed.
    promote_run_repeat = subprocess.run(promote_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert promote_run_repeat.returncode == 0, promote_run_repeat.stdout + promote_run_repeat.stderr
