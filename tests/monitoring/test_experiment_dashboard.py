import json
from pathlib import Path

import pandas as pd

from monitoring.experiment_dashboard import (
    _compare_metrics,
    _extract_regime_table,
    load_experiment_summaries,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_experiment_summaries_basic(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QUANTO_DASHBOARD_DISABLE_CACHE", "1")
    registry_root = tmp_path / "experiments"
    exp_dir = registry_root / "exp_1"
    _write_json(
        exp_dir / "spec" / "experiment_spec.json",
        {
            "experiment_name": "core_v1",
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "interval": "daily",
            "feature_set": "core_v1_regime",
            "policy": "ppo",
            "policy_params": {"reward_version": "reward_v2"},
        },
    )
    _write_json(
        exp_dir / "evaluation" / "metrics.json",
        {
            "performance": {"sharpe": 1.23, "total_return": 0.5, "max_drawdown": 0.2},
            "trading": {"turnover_1d_mean": 0.12, "tx_cost_bps": 5.0},
        },
    )
    _write_json(exp_dir / "logs" / "run_summary.json", {"recorded_at": "2024-01-02T00:00:00+00:00"})
    _write_json(exp_dir / "promotion" / "qualification_report.json", {"passed": True})

    missing_dir = registry_root / "exp_missing"
    _write_json(missing_dir / "spec" / "experiment_spec.json", {"experiment_name": "core_v1"})

    summaries = load_experiment_summaries(registry_root)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.experiment_id == "exp_1"
    assert summary.reward_version == "reward_v2"
    assert summary.sharpe == 1.23
    assert summary.qualification_passed is True


def test_extract_regime_table() -> None:
    payload = {
        "high_vol": {"sharpe": 1.1, "total_return": 0.2},
        "low_vol": {"sharpe": 2.2, "total_return": 0.4},
    }
    table = _extract_regime_table(payload)
    assert isinstance(table, pd.DataFrame)
    assert table.loc["high_vol", "sharpe"] == 1.1
    assert table.loc["low_vol", "total_return"] == 0.4


def test_compare_metrics_delta() -> None:
    candidate = {"performance": {"sharpe": 2.0}}
    baseline = {"performance": {"sharpe": 1.0}}
    table = _compare_metrics(candidate, baseline, [("Sharpe", "performance.sharpe")])
    row = table.iloc[0]
    assert row["delta"] == 1.0
    assert row["delta_pct"] == 100.0
