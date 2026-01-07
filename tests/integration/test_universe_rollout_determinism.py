from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from research.datasets.canonical_equity_loader import build_union_calendar, load_canonical_equity
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.features.feature_eng import build_sma_feature_result
from research.features.feature_registry import build_universe_feature_panel
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import run_rollout

UTC = timezone.utc


def _write_yearly_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _build_rows(symbol: str, closes: list[float]) -> list[dict]:
    rows: list[dict] = []
    for idx, close in enumerate(closes, start=1):
        ts = datetime(2023, 1, idx + 2, 16, tzinfo=UTC)
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts.isoformat(),
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000 + idx,
            }
        )
    return rows


def test_universe_rollout_determinism(tmp_path: Path):
    data_root = tmp_path / "quanto_data"
    symbols = ["AAPL", "MSFT"]
    values = {
        "AAPL": [150.0, 151.5, 152.0, 152.8, 153.1, 154.0],
        "MSFT": [240.0, 241.0, 241.5, 242.0, 242.2, 243.0],
    }
    for symbol in symbols:
        yearly_path = data_root / "canonical" / "equity_ohlcv" / symbol / "daily" / "2023.parquet"
        _write_yearly_file(yearly_path, _build_rows(symbol, values[symbol]))

    start = date(2023, 1, 4)
    end = date(2023, 1, 11)
    slices, canonical_hashes = load_canonical_equity(symbols, start, end, data_root=data_root)
    feature_map = {}
    for symbol in symbols:
        feature_map[symbol] = build_sma_feature_result(
            slices[symbol],
            fast_window=2,
            slow_window=3,
            feature_set="sma_v1",
            start_date=start,
            end_date=end,
            data_root=data_root,
        )
    calendar = build_union_calendar(slices, start_date=start, end_date=end)
    panel = build_universe_feature_panel(feature_map, symbol_order=symbols, calendar=calendar)

    def _run_once():
        env = SignalWeightTradingEnv(
            panel.rows,
            config=SignalWeightEnvConfig(transaction_cost_bp=0.5),
            observation_columns=panel.observation_columns,
        )
        policy = SMAWeightPolicy(SMAWeightPolicyConfig(mode="sigmoid", sigmoid_scale=3.0))
        hashes = dict(canonical_hashes)
        for symbol, result in feature_map.items():
            for key, value in result.inputs_used.items():
                hashes[f"{symbol}:{key}"] = value
        return run_rollout(env, policy, inputs_used=hashes)

    first = _run_once()
    second = _run_once()

    assert first.account_values == second.account_values
    assert first.weights == second.weights
    assert first.log_returns == second.log_returns
    assert first.steps == second.steps
    assert first.symbols == tuple(symbols)
    assert len(first.steps) + 1 == len(first.account_values)
    assert len(first.weights) == len(first.account_values)
    assert isinstance(first.steps[0]["weight_target"], dict)
