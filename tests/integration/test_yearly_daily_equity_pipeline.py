from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from research.datasets.canonical_equity_loader import load_canonical_equity
from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.features.feature_eng import build_sma_feature_result
from research.policies.sma_weight_policy import SMAWeightPolicy, SMAWeightPolicyConfig
from research.runners.rollout import run_rollout

UTC = timezone.utc


def _write_yearly_file(path: Path, closes: list[float]) -> None:
    rows = []
    for idx, close in enumerate(closes, start=1):
        ts = datetime(2023, 1, idx + 1, 16, tzinfo=UTC)
        rows.append(
            {
                "symbol": "AAPL",
                "timestamp": ts.isoformat(),
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1_000_000 + idx,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_yearly_daily_pipeline_deterministic(tmp_path: Path):
    data_root = tmp_path / "quanto_data"
    yearly_path = data_root / "canonical" / "equity_ohlcv" / "AAPL" / "daily" / "2023.parquet"
    _write_yearly_file(yearly_path, [100.0, 101.5, 102.0, 101.0, 103.0, 104.5])

    start = date(2023, 1, 3)
    end = date(2023, 1, 7)
    outputs = []
    for _ in range(2):
        slices, hashes = load_canonical_equity(["AAPL"], start, end, data_root=data_root)
        slice_data = slices["AAPL"]
        feature_result = build_sma_feature_result(
            slice_data,
            fast_window=2,
            slow_window=3,
            feature_set="sma_v1",
            start_date=start,
            end_date=end,
            data_root=data_root,
        )
        rows = feature_result.frame.to_dict("records")
        env = SignalWeightTradingEnv(
            rows,
            config=SignalWeightEnvConfig(transaction_cost_bp=0.5),
            observation_columns=feature_result.observation_columns,
        )
        policy = SMAWeightPolicy(SMAWeightPolicyConfig(mode="hard"))
        combined_hashes = dict(hashes)
        combined_hashes.update(feature_result.inputs_used)
        result = run_rollout(env, policy, inputs_used=combined_hashes)
        outputs.append(result)

    first, second = outputs
    assert first.account_values == second.account_values
    assert first.weights == second.weights
    assert first.metrics == second.metrics
    assert list(first.inputs_used.keys()) == sorted(first.inputs_used.keys())
    assert "canonical/equity_ohlcv/AAPL/daily/2023.parquet" in first.inputs_used
    assert first.timestamps[0] >= "2023-01-03"
    assert first.timestamps[-1] <= "2023-01-07T23:59:59"


def test_yearly_loader_missing_file_breaks_pipeline(tmp_path: Path):
    data_root = tmp_path / "quanto_data"
    with pytest.raises(FileNotFoundError):
        load_canonical_equity(["AAPL"], date(2023, 1, 2), date(2023, 1, 5), data_root=data_root)
