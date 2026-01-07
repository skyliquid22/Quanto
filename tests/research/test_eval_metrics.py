from __future__ import annotations

from research.eval.metrics import MetricConfig, compute_metric_bundle


def test_metric_bundle_expected_values():
    account_values = [100.0, 110.0, 105.0, 120.0]
    weights = [
        {"AAA": 0.5, "BBB": 0.5},
        {"AAA": 0.7, "BBB": 0.3},
        {"AAA": 0.4, "BBB": 0.6},
        {"AAA": 0.4, "BBB": 0.6},
    ]
    costs = [1.0, 0.5, 0.0]
    config = MetricConfig(float_precision=6)
    result = compute_metric_bundle(
        account_values,
        weights,
        transaction_costs=costs,
        symbols=("AAA", "BBB"),
        config=config,
    )
    assert result.performance["total_return"] == 0.2
    assert result.performance["volatility_ann"] == 1.279352
    assert result.performance["sharpe"] == 12.961107
    assert result.performance["max_drawdown"] == 0.045455
    assert result.performance["cagr"] is None
    assert result.trading["turnover_1d_mean"] == 0.333333
    assert result.trading["turnover_1d_median"] == 0.4
    assert result.trading["avg_exposure"] == 1.0
    assert result.trading["max_concentration"] == 0.6
    assert result.trading["hhi_mean"] == 0.53
    assert result.trading["tx_cost_total"] == 1.5
    assert result.trading["tx_cost_bps"] == 137.931034
    assert result.safety["nan_inf_violations"] == 0.0
    assert result.safety["action_bounds_violations"] == 0.0
    assert result.returns == [0.1, -0.045455, 0.142857]


def test_metric_bundle_nan_detection():
    account_values = [100.0, float("nan")]
    weights = [
        {"AAA": 0.2},
        {"AAA": float("inf"), "BBB": -0.5},
    ]
    costs = [float("nan")]
    result = compute_metric_bundle(
        account_values,
        weights,
        transaction_costs=costs,
        symbols=("AAA", "BBB"),
    )
    assert result.safety["nan_inf_violations"] == 4.0
    assert result.safety["action_bounds_violations"] == 1.0
