from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3", reason="PPO smoke test requires stable-baselines3")

from research.datasets.canonical_equity_loader import load_canonical_equity
from research.envs.gym_weight_env import GymWeightTradingEnv
from research.envs.signal_weight_env import SignalWeightEnvConfig
from research.features.feature_eng import build_sma_feature_result
from research.training.ppo_trainer import evaluate, train_ppo

UTC = timezone.utc


def _write_yearly_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _build_rows(timestamps: list[datetime], base: float) -> list[dict]:
    entries: list[dict] = []
    for idx, ts in enumerate(timestamps):
        close = base + idx * 0.3
        entries.append(
            {
                "symbol": "SPY",
                "timestamp": ts.isoformat(),
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 1.0,
                "close": close,
                "volume": 10_000_000 + idx,
            }
        )
    return entries


def _days(year: int, month: int, first: int, last: int) -> list[datetime]:
    return [datetime(year, month, day, 16, tzinfo=UTC) for day in range(first, last + 1)]


def test_ppo_smoke_yearly_daily(tmp_path: Path):
    data_root = tmp_path / "quanto_data"
    base = data_root / "canonical" / "equity_ohlcv" / "SPY" / "daily"
    dec_days = _days(2022, 12, 20, 31)
    jan_days = _days(2023, 1, 2, 15)
    _write_yearly_file(base / "2022.parquet", _build_rows(dec_days, 400.0))
    _write_yearly_file(base / "2023.parquet", _build_rows(jan_days, 403.5))

    start = date(2022, 12, 22)
    end = date(2023, 1, 12)
    slices, _ = load_canonical_equity(["SPY"], start, end, data_root=data_root)
    slice_data = slices["SPY"]
    feature_result = build_sma_feature_result(
        slice_data,
        fast_window=2,
        slow_window=5,
        feature_set="sma_v1",
        start_date=start,
        end_date=end,
        data_root=data_root,
    )
    rows = feature_result.frame.to_dict("records")
    env_config = SignalWeightEnvConfig(transaction_cost_bp=0.1)
    train_env = GymWeightTradingEnv(rows, config=env_config, observation_columns=feature_result.observation_columns)
    eval_env = GymWeightTradingEnv(rows, config=env_config, observation_columns=feature_result.observation_columns)

    model = train_ppo(
        train_env,
        total_timesteps=256,
        seed=7,
        learning_rate=5e-4,
        gamma=0.95,
    )
    result = evaluate(model, eval_env)

    assert len(result.timestamps) == len(result.account_values)
    assert len(result.log_returns) == len(result.steps)
    assert result.metrics["num_steps"] == len(result.log_returns)
    assert min(result.account_values) > 0.0
    assert result.timestamps[0].startswith("2022-12-")
    assert result.timestamps[-1] >= "2023-01-10"
    for value in result.metrics.values():
        assert math.isfinite(value)
