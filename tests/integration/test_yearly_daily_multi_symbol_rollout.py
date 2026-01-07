from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from research.datasets.canonical_equity_loader import load_canonical_equity
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.features.feature_eng import build_sma_feature_result
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import run_rollout

UTC = timezone.utc


def _write_yearly_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _build_rows(symbol: str, start_day: int, closes: list[float]) -> list[dict]:
    rows: list[dict] = []
    for idx, close in enumerate(closes):
        ts = datetime(2023, 1, start_day + idx, 16, tzinfo=UTC)
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts.isoformat(),
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 5_000_000 + idx,
            }
        )
    return rows


def _run_rollout_for_symbol(symbol: str, start: date, end: date, data_root: Path):
    slices, canonical_hashes = load_canonical_equity([symbol], start, end, data_root=data_root)
    slice_data = slices[symbol]
    feature_result = build_sma_feature_result(
        slice_data,
        fast_window=2,
        slow_window=4,
        feature_set="sma_v1",
        start_date=start,
        end_date=end,
        data_root=data_root,
    )
    rows = feature_result.frame.to_dict("records")
    env = SignalWeightTradingEnv(
        rows,
        config=SignalWeightEnvConfig(transaction_cost_bp=0.25),
        observation_columns=feature_result.observation_columns,
    )
    policy = SMAWeightPolicy(SMAWeightPolicyConfig(mode="sigmoid", sigmoid_scale=4.0))
    combined_hashes = dict(canonical_hashes)
    combined_hashes.update(feature_result.inputs_used)
    return run_rollout(env, policy, inputs_used=combined_hashes)


def test_multi_symbol_rollouts_are_deterministic(tmp_path: Path):
    data_root = tmp_path / "quanto_data"
    base = data_root / "canonical" / "equity_ohlcv"
    aapl_rows = _build_rows("AAPL", start_day=2, closes=[150.0, 151.0, 152.5, 153.0, 152.0, 151.5, 152.2, 153.4, 154.1])
    msft_rows = _build_rows("MSFT", start_day=2, closes=[240.0, 240.5, 241.0, 241.5, 242.0, 241.2, 240.8, 241.6, 242.3])
    _write_yearly_file(base / "AAPL" / "daily" / "2023.parquet", aapl_rows)
    _write_yearly_file(base / "MSFT" / "daily" / "2023.parquet", msft_rows)

    start = date(2023, 1, 3)
    end = date(2023, 1, 10)
    symbols = ("AAPL", "MSFT")
    runs = {symbol: (_run_rollout_for_symbol(symbol, start, end, data_root), _run_rollout_for_symbol(symbol, start, end, data_root)) for symbol in symbols}

    for symbol, (first, second) in runs.items():
        assert first.account_values == second.account_values
        assert first.weights == second.weights
        assert first.log_returns == second.log_returns
        assert first.steps == second.steps
        assert first.metrics == second.metrics
        assert first.timestamps == second.timestamps
        assert len(first.timestamps) == len(first.account_values)
        assert len(first.log_returns) == len(first.steps)
        assert first.metrics["num_steps"] == len(first.log_returns)
        rel_path = f"canonical/equity_ohlcv/{symbol}/daily/2023.parquet"
        assert rel_path in first.inputs_used
        assert first.timestamps[0][:10] >= start.isoformat()
        assert first.timestamps[-1][:10] <= end.isoformat()

    aapl_first = runs["AAPL"][0]
    msft_first = runs["MSFT"][0]
    assert aapl_first.timestamps == msft_first.timestamps
    assert len(aapl_first.steps) == len(msft_first.steps)
