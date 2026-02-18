from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.eval.regime_slicing import compute_regime_slices


def _write_thresholds(path: Path) -> Path:
    payload = {
        "version": "v2",
        "universe": ["SPY", "QQQ", "IWM"],
        "reference_window": {"start_date": "2005-01-01", "end_date": "2025-12-31"},
        "thresholds": {
            "market_vol_20d": {"high": 0.012},
            "market_trend_20d": {"deadzone": 0.002},
        },
        "bucket_distribution": {
            "high_vol_trend_up": 0.1,
            "high_vol_trend_down": 0.1,
            "high_vol_flat": 0.1,
            "low_vol_trend_up": 0.2,
            "low_vol_trend_down": 0.2,
            "low_vol_flat": 0.3,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_regime_v2_bucket_mapping(tmp_path: Path) -> None:
    thresholds_path = _write_thresholds(tmp_path / "thresholds.json")
    feature_names = ("market_vol_20d", "market_trend_20d")
    regime_series = [
        [0.015, 0.001],  # high_vol_flat
        [0.015, 0.010],  # high_vol_trend_up
        [0.005, -0.003],  # low_vol_trend_down
        [0.005, 0.0],  # low_vol_flat
    ]
    returns = [0.01, 0.02, 0.03]
    exposures = [0.1, 0.2, 0.3, 0.4]
    turnover = [0.01, 0.02, 0.03, 0.04]
    result = compute_regime_slices(
        regime_series,
        feature_names,
        returns=returns,
        exposures=exposures,
        turnover_by_step=turnover,
        annualization_days=252,
        float_precision=6,
        labeling_version="v2",
        thresholds_path=thresholds_path,
    )
    assert result is not None
    perf = result.performance_by_regime
    assert perf["high_vol_flat"]["avg_exposure"] == 0.1
    assert perf["high_vol_trend_up"]["avg_exposure"] == 0.2
    assert perf["low_vol_trend_down"]["avg_exposure"] == 0.3
    assert perf["low_vol_flat"]["avg_exposure"] == 0.4
    assert result.metadata["thresholds_used"]["market_vol_20d_high"] == 0.012


def test_regime_v2_requires_thresholds(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_thresholds.yml"
    with pytest.raises(FileNotFoundError):
        compute_regime_slices(
            [[0.1, 0.0]],
            ("market_vol_20d", "market_trend_20d"),
            returns=[0.0],
            exposures=[0.1],
            turnover_by_step=[0.0],
            annualization_days=252,
            float_precision=6,
            labeling_version="v2",
            thresholds_path=missing_path,
        )
