from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_experiment(
    registry_root: Path,
    experiment_id: str,
    experiment_name: str,
    *,
    sharpe: float,
    max_drawdown: float,
    turnover: float,
    nan_inf: float = 0.0,
    recorded_at: datetime | None = None,
) -> None:
    base = registry_root / experiment_id
    (base / "spec").mkdir(parents=True, exist_ok=True)
    (base / "evaluation").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    spec_payload = {"experiment_name": experiment_name}
    (base / "spec" / "experiment_spec.json").write_text(json.dumps(spec_payload), encoding="utf-8")
    metrics_payload = _metrics_payload(
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        turnover=turnover,
        nan_inf=nan_inf,
        experiment_id=experiment_id,
    )
    (base / "evaluation" / "metrics.json").write_text(
        json.dumps(metrics_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    recorded = recorded_at or datetime.now(tz=timezone.utc)
    (base / "logs" / "run_summary.json").write_text(
        json.dumps({"recorded_at": recorded.isoformat()}),
        encoding="utf-8",
    )


def _metrics_payload(
    *,
    sharpe: float,
    max_drawdown: float,
    turnover: float,
    nan_inf: float,
    experiment_id: str,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "symbols": ["AAA", "BBB"],
            "start_date": "2023-01-01",
            "end_date": "2023-06-30",
            "interval": "daily",
            "feature_set": "baseline_feature_set",
            "policy_id": "equal_weight",
            "run_id": experiment_id,
        },
        "performance": {
            "total_return": 0.12,
            "cagr": None,
            "volatility_ann": 0.3,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": 1.2,
        },
        "trading": {
            "turnover_1d_mean": turnover,
            "turnover_1d_median": turnover,
            "avg_exposure": 1.0,
            "max_concentration": 0.5,
            "hhi_mean": 0.4,
            "tx_cost_total": 4.0,
            "tx_cost_bps": 12.0,
        },
        "safety": {
            "nan_inf_violations": nan_inf,
            "action_bounds_violations": 0.0,
        },
    }


def test_regression_gate_cli_enforcement(tmp_path: Path):
    registry_root = tmp_path / "experiments"
    registry_root.mkdir()
    baseline_name = "equal_weight_universe_v1"
    _write_experiment(
        registry_root,
        experiment_id="baseline_exp",
        experiment_name=baseline_name,
        sharpe=1.0,
        max_drawdown=0.08,
        turnover=0.05,
    )
    _write_experiment(
        registry_root,
        experiment_id="candidate_hard_fail",
        experiment_name="candidate_strategy",
        sharpe=0.8,
        max_drawdown=0.09,
        turnover=0.06,
    )
    _write_experiment(
        registry_root,
        experiment_id="candidate_soft_warn",
        experiment_name="candidate_strategy",
        sharpe=0.97,
        max_drawdown=0.1,
        turnover=0.05,
    )
    gate_config = {
        "gates": [
            {"id": "sharpe_guard", "metric": "performance.sharpe", "type": "hard", "threshold_pct": 5.0},
            {"id": "drawdown_guard", "metric": "performance.max_drawdown", "type": "soft", "threshold_pct": 5.0},
            {"id": "turnover_guard", "metric": "trading.turnover_1d_mean", "type": "soft", "threshold_pct": 5.0},
            {"id": "nan_guard", "metric": "safety.nan_inf_violations", "type": "hard", "max_value": 0.0},
        ]
    }
    gate_path = tmp_path / "gate_config.json"
    gate_path.write_text(json.dumps(gate_config, indent=2), encoding="utf-8")

    hard_cmd = [
        sys.executable,
        "scripts/compare_experiments.py",
        "--candidate",
        "candidate_hard_fail",
        "--baseline",
        baseline_name,
        "--gates",
        str(gate_path),
        "--registry-root",
        str(registry_root),
    ]
    hard_run = subprocess.run(hard_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert hard_run.returncode == 1, hard_run.stdout + hard_run.stderr
    hard_report_path = registry_root / "candidate_hard_fail" / "comparison" / "gate_report.json"
    assert hard_report_path.exists()
    hard_report = json.loads(hard_report_path.read_text(encoding="utf-8"))
    assert hard_report["overall_status"] == "fail"
    assert hard_report["hard_failures"] >= 1

    soft_cmd = [
        sys.executable,
        "scripts/compare_experiments.py",
        "--candidate",
        "candidate_soft_warn",
        "--baseline",
        baseline_name,
        "--gates",
        str(gate_path),
        "--registry-root",
        str(registry_root),
    ]
    soft_run = subprocess.run(soft_cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert soft_run.returncode == 0, soft_run.stdout + soft_run.stderr
    soft_report_path = registry_root / "candidate_soft_warn" / "comparison" / "gate_report.json"
    assert soft_report_path.exists()
    soft_report = json.loads(soft_report_path.read_text(encoding="utf-8"))
    assert soft_report["overall_status"] == "pass"
    assert soft_report["soft_warnings"] >= 1
