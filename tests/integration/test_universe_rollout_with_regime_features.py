from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.eval.metrics import MetricConfig, compute_metric_bundle
from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.features import feature_registry
from research.features.feature_registry import FeatureSetResult, SMA_OBSERVATION_COLUMNS, build_universe_feature_panel
from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS
from research.regime import RegimeState


def _build_feature_frame(closes: Sequence[float], start: str) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    data = {
        "timestamp": timestamps,
        "close": closes,
        "sma_fast": closes,
        "sma_slow": closes,
        "sma_diff": [a - b for a, b in zip(closes, closes)],
        "sma_signal": [0.0 for _ in closes],
    }
    return pd.DataFrame(data)


def _feature_result(frame: pd.DataFrame) -> FeatureSetResult:
    return FeatureSetResult(
        frame=frame,
        observation_columns=SMA_OBSERVATION_COLUMNS,
        feature_set="sma_universe_v1",
        inputs_used={},
    )


def _make_primary_slice(symbol: str, closes: Sequence[float]) -> CanonicalEquitySlice:
    timestamps = pd.date_range("2023-01-01", periods=len(closes), freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000_000 for _ in closes],
        },
        index=timestamps,
    )
    frame.index.name = "timestamp"
    return CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=[])


def _run_constant_rollout(panel_rows: Sequence[Dict[str, object]], observation_columns: Sequence[str]):
    env = SignalWeightTradingEnv(panel_rows, config=SignalWeightEnvConfig(), observation_columns=observation_columns)
    env.reset()
    symbols = env.symbols
    weights: List[Dict[str, float]] = [{symbol: 0.0 for symbol in symbols}]
    account_values = [env.portfolio_value]
    costs: List[float] = []
    regime_series: List[Tuple[float, ...]] | None = None
    regime_names: Tuple[str, ...] | None = None
    initial_state = env.current_row.get("regime_state")
    if isinstance(initial_state, RegimeState):
        regime_series = [tuple(float(value) for value in initial_state.features)]
        regime_names = initial_state.feature_names
    done = False
    while not done:
        _, _, done, info = env.step([0.5] * env.num_assets)
        account_values.append(float(info["portfolio_value"]))
        realized = info["weight_realized"]
        weights.append({symbol: float(realized[symbol]) for symbol in symbols})
        costs.append(float(info["cost_paid"]))
        next_state = env.current_row.get("regime_state")
        if regime_names:
            assert isinstance(next_state, RegimeState)
            regime_series = regime_series or []
            regime_series.append(tuple(float(value) for value in next_state.features))
    return account_values, weights, costs, regime_names, regime_series, symbols


def test_universe_rollout_with_regime_features():
    symbols = ("AAA", "BBB")
    closes_a = [100.0, 101.0, 103.0, 102.0, 104.0]
    closes_b = [50.0, 51.0, 50.5, 52.0, 53.0]
    frame_a = _build_feature_frame(closes_a, "2023-01-01")
    frame_b = _build_feature_frame(closes_b, "2023-01-01")

    feature_map = {
        "AAA": _feature_result(frame_a),
        "BBB": _feature_result(frame_b),
    }

    calendar = frame_a["timestamp"]
    baseline_panel = build_universe_feature_panel(
        feature_map,
        symbol_order=symbols,
        calendar=calendar,
    )
    regime_panel = build_universe_feature_panel(
        feature_map,
        symbol_order=symbols,
        calendar=calendar,
        regime_feature_set="regime_v1",
    )

    assert baseline_panel.observation_columns == SMA_OBSERVATION_COLUMNS
    assert regime_panel.observation_columns == SMA_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS
    assert all("regime_state" in row for row in regime_panel.rows)
    first_state = regime_panel.rows[0]["regime_state"]
    assert isinstance(first_state, RegimeState)
    assert first_state.feature_names == REGIME_FEATURE_COLUMNS

    cfg = MetricConfig(risk_config=SignalWeightEnvConfig().risk_config)
    base_account, base_weights, base_costs, _, _, _ = _run_constant_rollout(
        baseline_panel.rows, baseline_panel.observation_columns
    )
    (
        regime_account,
        regime_weights,
        regime_costs,
        regime_names,
        regime_series,
        _
    ) = _run_constant_rollout(regime_panel.rows, regime_panel.observation_columns)

    assert regime_names == REGIME_FEATURE_COLUMNS
    assert regime_series is not None
    assert len(regime_series) == len(regime_weights)

    base_metrics = compute_metric_bundle(
        base_account,
        base_weights,
        transaction_costs=base_costs,
        symbols=symbols,
        config=cfg,
    )
    regime_metrics = compute_metric_bundle(
        regime_account,
        regime_weights,
        transaction_costs=regime_costs,
        symbols=symbols,
        config=cfg,
        regime_feature_series=regime_series,
        regime_feature_names=regime_names,
    )

    assert base_metrics.performance == regime_metrics.performance
    assert base_metrics.safety == regime_metrics.safety

    diag_keys = {"avg_exposure_by_regime", "avg_turnover_by_regime", "regime_feature_summary"}
    for key, value in base_metrics.trading.items():
        if key in diag_keys:
            assert value == {}
            assert regime_metrics.trading[key] != {}
        else:
            assert regime_metrics.trading[key] == value
    for key in diag_keys:
        assert key in regime_metrics.trading
        assert regime_metrics.trading[key]


def test_universe_rollout_with_primary_regime_features(monkeypatch):
    symbols = ("AAA", "BBB")
    closes_a = [100.0, 101.0, 103.0, 102.0, 104.0]
    closes_b = [50.0, 51.0, 50.5, 52.0, 53.0]
    frame_a = _build_feature_frame(closes_a, "2023-01-01")
    frame_b = _build_feature_frame(closes_b, "2023-01-01")

    primary_slices = {
        "SPY": _make_primary_slice("SPY", [300.0, 301.0, 299.0, 302.0, 303.0]),
        "QQQ": _make_primary_slice("QQQ", [250.0, 251.0, 252.0, 253.0, 254.0]),
        "IWM": _make_primary_slice("IWM", [180.0, 181.0, 182.0, 183.0, 184.0]),
    }

    def fake_load_primary(start_date, end_date, *, data_root=None, interval="daily"):
        assert interval == "daily"
        return dict(primary_slices), {}

    monkeypatch.setattr(feature_registry, "load_primary_regime_universe", fake_load_primary)

    feature_map = {
        "AAA": _feature_result(frame_a),
        "BBB": _feature_result(frame_b),
    }

    calendar = frame_a["timestamp"]
    regime_panel = build_universe_feature_panel(
        feature_map,
        symbol_order=symbols,
        calendar=calendar,
        regime_feature_set="regime_v1_1",
    )

    assert regime_panel.observation_columns == SMA_OBSERVATION_COLUMNS + REGIME_FEATURE_COLUMNS
    assert all("regime_state" in row for row in regime_panel.rows)
    first_state = regime_panel.rows[0]["regime_state"]
    assert isinstance(first_state, RegimeState)
    assert first_state.feature_names == REGIME_FEATURE_COLUMNS
    cfg = MetricConfig(risk_config=SignalWeightEnvConfig().risk_config)
    (
        account_values,
        weights,
        costs,
        regime_names,
        regime_series,
        _,
    ) = _run_constant_rollout(regime_panel.rows, regime_panel.observation_columns)

    metrics = compute_metric_bundle(
        account_values,
        weights,
        transaction_costs=costs,
        symbols=symbols,
        config=cfg,
        regime_feature_series=regime_series,
        regime_feature_names=regime_names,
    )
    assert metrics.trading["regime_feature_summary"]
