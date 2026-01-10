from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_regime_slices


def _write_rollout(path: Path, *, include_regime: bool) -> None:
    payload = {
        "metadata": {
            "symbols": ["AAA", "BBB"],
            "rollout": {"symbols": ["AAA", "BBB"]},
        },
        "series": {
            "timestamps": [
                "2023-01-02T00:00:00+00:00",
                "2023-01-03T00:00:00+00:00",
                "2023-01-04T00:00:00+00:00",
            ],
            "returns": [0.01, -0.005],
            "weights": {"AAA": [0.0, 0.6, 0.4], "BBB": [0.0, 0.4, 0.6]},
        },
    }
    if include_regime:
        payload["series"]["regime"] = {
            "feature_names": ["market_vol_20d", "market_trend_20d"],
            "values": [[0.2, 0.1], [0.3, 0.05]],
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_regime_slices_creates_timeseries_and_regime(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    exp_dir = data_root / "experiments" / "exp123"
    (exp_dir / "runs").mkdir(parents=True)
    (exp_dir / "evaluation").mkdir(parents=True)
    _write_rollout(exp_dir / "runs" / "rollout.json", include_regime=True)
    result = run_regime_slices.main(
        ["--experiment-id", "exp123", "--data-root", str(data_root)]
    )
    assert result == 0
    ts_path = exp_dir / "evaluation" / "timeseries.json"
    slices_path = exp_dir / "evaluation" / "regime_slices.json"
    assert ts_path.exists()
    assert slices_path.exists()
    regime_slices = json.loads(slices_path.read_text(encoding="utf-8"))
    assert "performance_by_regime" in regime_slices
    assert set(regime_slices["performance_by_regime"]) == {"low_vol", "mid_vol", "high_vol"}


def test_run_regime_slices_without_regime_series(tmp_path):
    data_root = tmp_path / "data"
    exp_dir = data_root / "experiments" / "exp456"
    (exp_dir / "runs").mkdir(parents=True)
    (exp_dir / "evaluation").mkdir(parents=True)
    _write_rollout(exp_dir / "runs" / "rollout.json", include_regime=False)
    result = run_regime_slices.main(
        ["--experiment-id", "exp456", "--data-root", str(data_root)]
    )
    assert result == 0
    ts_path = exp_dir / "evaluation" / "timeseries.json"
    slices_path = exp_dir / "evaluation" / "regime_slices.json"
    assert ts_path.exists()
    assert not slices_path.exists()
