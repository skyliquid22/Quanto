from __future__ import annotations

from research.eval.regime_slicing import compute_regime_slices


def test_regime_slicing_metrics_by_bucket():
    regime_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    regime_series = [[value] for value in regime_values]
    returns = [0.01, -0.02, 0.03, -0.01, 0.02]
    exposures = [0.1, 0.15, 0.4, 0.5, 0.8, 0.9]
    turnover = [0.0, 0.02, 0.04, 0.01, 0.03, 0.05]
    result = compute_regime_slices(
        regime_series,
        ("market_vol_20d",),
        returns=returns,
        exposures=exposures,
        turnover_by_step=turnover,
        annualization_days=252,
        float_precision=6,
    )
    assert result is not None
    assert result.metadata["signal"] == "market_vol_20d"
    quantiles = result.metadata["quantiles"]
    assert quantiles["q33"] == 0.265
    assert quantiles["q66"] == 0.56
    low_metrics = result.performance_by_regime["low_vol"]
    high_metrics = result.performance_by_regime["high_vol"]
    assert low_metrics["avg_exposure"] == 0.125
    assert high_metrics["avg_exposure"] == 0.85
    assert high_metrics["avg_turnover"] == 0.04
    assert high_metrics["total_return"] == 0.02
    assert high_metrics["max_drawdown"] == 0.0


def test_regime_slicing_handles_duplicate_values():
    regime_series = [[0.5]] * 4
    returns = [0.01, 0.01, 0.01]
    exposures = [0.2, 0.2, 0.2, 0.2]
    turnover = [0.0, 0.01, 0.01, 0.01]
    result = compute_regime_slices(
        regime_series,
        ("market_vol_20d",),
        returns=returns,
        exposures=exposures,
        turnover_by_step=turnover,
        annualization_days=252,
        float_precision=6,
    )
    assert result is not None
    quantiles = result.metadata["quantiles"]
    assert quantiles["q33"] == 0.5
    assert quantiles["q66"] == 0.5
    assert result.performance_by_regime["mid_vol"]["avg_exposure"] is None
    assert result.performance_by_regime["high_vol"]["avg_exposure"] is None
    low_metrics = result.performance_by_regime["low_vol"]
    assert low_metrics["avg_exposure"] == 0.2
    assert low_metrics["total_return"] > 0
