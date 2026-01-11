from __future__ import annotations

import pytest

from research.eval.evaluate import EvalSeries, _series_section
from research.eval.metrics import MetricResult


def _metric_result(returns):
    return MetricResult(
        performance={},
        trading={},
        safety={},
        returns=list(returns),
    )


def test_series_section_includes_regime_block():
    series = EvalSeries(
        timestamps=["t0", "t1", "t2"],
        account_values=[100.0, 101.0, 102.0],
        weights=[
            {"AAA": 0.0},
            {"AAA": 1.0},
            {"AAA": 1.0},
        ],
        transaction_costs=[0.0, 0.0],
        symbols=("AAA",),
        regime_features=[[0.1, 0.2], [0.3, 0.4]],
        regime_feature_names=("market_vol_20d", "market_trend_20d"),
    )
    metrics = _metric_result([0.01, 0.02])
    payload = _series_section(series, metrics)
    assert "regime" in payload
    assert payload["regime"]["feature_names"] == ["market_vol_20d", "market_trend_20d"]
    assert payload["regime"]["values"] == [[0.1, 0.2], [0.3, 0.4]]


def test_series_section_validates_alignment():
    series = EvalSeries(
        timestamps=["t0", "t1", "t2"],
        account_values=[100.0, 101.0, 102.0],
        weights=[
            {"AAA": 0.0},
            {"AAA": 1.0},
            {"AAA": 1.0},
        ],
        transaction_costs=[0.0, 0.0],
        symbols=("AAA",),
        regime_features=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        regime_feature_names=("market_vol_20d", "market_trend_20d"),
    )
    metrics = _metric_result([0.01, 0.02])
    with pytest.raises(ValueError):
        _series_section(series, metrics)
