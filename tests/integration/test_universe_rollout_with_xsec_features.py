from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from research.datasets.canonical_equity_loader import build_union_calendar, load_canonical_equity
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.features.feature_eng import build_universe_feature_results
from research.features.feature_registry import build_universe_feature_panel
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import run_rollout

UTC = timezone.utc


def _write_yearly(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _rows(symbol: str, closes: list[float]) -> list[dict]:
    entries: list[dict] = []
    for idx, close in enumerate(closes, start=1):
        ts = datetime(2022, 1, idx + 2, 16, tzinfo=UTC)
        entries.append(
            {
                "symbol": symbol,
                "timestamp": ts.isoformat(),
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": 1_000_000 + idx,
            }
        )
    return entries


def test_universe_rollout_with_cross_sectional_features(tmp_path: Path):
    data_root = tmp_path / "quanto_data"
    symbols = ["AAA", "BBB", "CCC"]
    price_map = {
        "AAA": [11.0, 11.5, 12.0, 12.4, 12.6, 12.9],
        "BBB": [40.0, 39.0, 39.4, 40.2, 41.0, 41.8],
        "CCC": [22.0, 22.5, 22.1, 22.0, 22.4, 22.9],
    }
    for symbol in symbols:
        path = data_root / "canonical" / "equity_ohlcv" / symbol / "daily" / "2022.parquet"
        _write_yearly(path, _rows(symbol, price_map[symbol]))

    start = date(2022, 1, 5)
    end = date(2022, 1, 12)
    slices, canonical_hashes = load_canonical_equity(symbols, start, end, data_root=data_root)
    feature_map = build_universe_feature_results(
        "equity_xsec_v1",
        slices,
        symbol_order=symbols,
        start_date=start,
        end_date=end,
    )
    calendar = build_union_calendar(slices, start_date=start, end_date=end)
    panel = build_universe_feature_panel(feature_map, symbol_order=symbols, calendar=calendar)
    env_config = SignalWeightEnvConfig(transaction_cost_bp=0.25)

    def _run_once():
        env = SignalWeightTradingEnv(panel.rows, config=env_config, observation_columns=panel.observation_columns)
        policy = SMAWeightPolicy(
            SMAWeightPolicyConfig(fast_key="ret_1d", slow_key="market_ret_1d", mode="sigmoid", sigmoid_scale=3.5)
        )
        inputs = dict(canonical_hashes)
        for symbol, result in feature_map.items():
            for key, value in result.inputs_used.items():
                inputs[f"{symbol}:{key}"] = value
        return run_rollout(env, policy, inputs_used=inputs)

    first = _run_once()
    second = _run_once()

    assert first.account_values == second.account_values
    assert first.weights == second.weights
    assert first.log_returns == second.log_returns
    assert first.symbols == tuple(symbols)
    assert len(first.steps) + 1 == len(first.account_values)
    assert all(isinstance(step["weight_target"], dict) for step in first.steps)
