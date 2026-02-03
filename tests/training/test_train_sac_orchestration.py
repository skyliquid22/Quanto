from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import pandas as pd

from research.features.feature_registry import FeatureSetResult
from research.training.ppo_trainer import EvaluationResult
from scripts import run_sma_finrl_rollout as rollout
from scripts import train_sac_weight_agent as trainer


UTC = timezone.utc


def _make_args(tmp_path, **overrides):
    defaults = {
        "symbol": "AAPL",
        "symbols": None,
        "start_date": "2023-01-02",
        "end_date": "2023-01-05",
        "interval": "daily",
        "fast_window": 2,
        "slow_window": 3,
        "transaction_cost_bp": 1.0,
        "seed": 7,
        "policy_preset": "sac_core_v1",
        "policy_params": {"timesteps": 1000},
        "reward_version": "reward_v1",
        "run_id": None,
        "data_root": str(tmp_path),
        "feature_set": "sma_v1",
        "regime_feature_set": None,
        "vendor": "polygon",
        "live": False,
        "ingest_mode": "rest",
        "canonical_domain": "equity_ohlcv",
        "force_ingest": False,
        "force_canonical_build": False,
        "max_weight": 1.0,
        "exposure_cap": 1.0,
        "min_cash": 0.0,
        "max_turnover_1d": None,
        "allow_short": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_common_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path, *, manifest):
    def fake_load(symbols, start, end, data_root=None, interval="daily"):
        entries = {}
        hashes = {}
        for symbol in symbols:
            slice_data = SimpleNamespace(
                rows=[{"timestamp": datetime(2023, 1, 2, tzinfo=UTC)}],
                timestamps=[datetime(2023, 1, 2, tzinfo=UTC), datetime(2023, 1, 3, tzinfo=UTC)],
                closes=[100.0, 101.0],
                symbol=symbol,
            )
            entries[symbol] = slice_data
            hashes[f"canonical/equity_ohlcv/{symbol}/daily/2023.parquet"] = "sha256:deadbeef"
        return entries, hashes

    sample_rows = [
        {
            "timestamp": datetime(2023, 1, 2, tzinfo=UTC),
            "close": 10.0,
            "sma_fast": 9.5,
            "sma_slow": 8.0,
            "sma_diff": 1.5,
            "sma_signal": 1.0,
        },
        {
            "timestamp": datetime(2023, 1, 3, tzinfo=UTC),
            "close": 11.0,
            "sma_fast": 9.7,
            "sma_slow": 8.5,
            "sma_diff": 1.2,
            "sma_signal": 1.0,
        },
    ]
    feature_rows = [*sample_rows, sample_rows[-1]]
    sample_frame = pd.DataFrame(feature_rows)

    class DummyModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.5], dtype=np.float32), None

        def save(self, path: str) -> None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("model", encoding="utf-8")

    eval_result = EvaluationResult(
        metrics={"total_return": 0.05},
        timestamps=["t0", "t1"],
        account_values=[10000.0, 10500.0],
        weights=[{"AAPL": 0.0}, {"AAPL": 0.5}],
        log_returns=[0.01],
        steps=[
            {
                "timestamp": "t0",
                "price_close": 10.0,
                "weight_target": 0.5,
                "weight_realized": 0.5,
                "portfolio_value": 10500.0,
                "cost_paid": 0.0,
                "reward": 0.01,
            }
        ],
        symbols=("AAPL",),
    )

    monkeypatch.setattr(trainer, "load_canonical_equity", fake_load)
    monkeypatch.setattr(trainer, "strategy_to_feature_frame", lambda strategy: sample_frame.copy())

    def fake_build_features(*args, **kwargs):
        return FeatureSetResult(
            frame=sample_frame.copy(),
            observation_columns=("close", "sma_fast", "sma_slow", "sma_diff", "sma_signal"),
            feature_set="sma_v1",
            inputs_used={},
        )

    monkeypatch.setattr(trainer, "build_features", fake_build_features)
    monkeypatch.setattr(trainer, "run_sma_crossover", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(trainer, "evaluate", lambda model, env: eval_result)
    monkeypatch.setattr(trainer, "ensure_yearly_daily_coverage", lambda **kwargs: {"missing_pairs": []})
    monkeypatch.setattr(trainer, "_locate_canonical_manifest", lambda *args, **kwargs: manifest)
    return DummyModel


def _create_manifest(tmp_path, args, run_id="canon"):
    manifest_path = Path(tmp_path) / "canonical" / "manifests" / args.canonical_domain / f"{run_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "domain": args.canonical_domain,
        "run_id": run_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "creation_timestamp": "2023-01-10T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return rollout.ManifestMetadata(
        path=manifest_path,
        run_id=run_id,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        created_at=datetime(2023, 1, 10, tzinfo=UTC),
        domain=args.canonical_domain,
        vendor=args.vendor,
    )


def test_interval_guard_raises(tmp_path):
    args = _make_args(tmp_path, interval="weekly")
    with pytest.raises(SystemExit):
        trainer.run_training(args)


def test_policy_params_override_preset(tmp_path, monkeypatch: pytest.MonkeyPatch):
    args = _make_args(
        tmp_path,
        policy_params={"timesteps": 123, "learning_rate": 1e-4, "buffer_size": 5000, "fast_window": 2, "slow_window": 4},
    )
    manifest = _create_manifest(tmp_path, args)
    DummyModel = _patch_common_dependencies(monkeypatch, tmp_path, manifest=manifest)

    captured: Dict[str, Any] = {}

    def fake_train(env, total_timesteps, learning_rate=None, **kwargs):
        captured["timesteps"] = total_timesteps
        captured["learning_rate"] = learning_rate
        captured["buffer_size"] = kwargs.get("buffer_size")
        return DummyModel()

    monkeypatch.setattr(trainer, "train_sac", fake_train)

    trainer.run_training(args)

    assert captured["timesteps"] == 123
    assert captured["learning_rate"] == pytest.approx(1e-4)
    assert captured["buffer_size"] == 5000


def test_missing_sb3_dependency_surfaces(tmp_path, monkeypatch: pytest.MonkeyPatch):
    args = _make_args(tmp_path)
    manifest = _create_manifest(tmp_path, args)
    _patch_common_dependencies(monkeypatch, tmp_path, manifest=manifest)

    def raise_runtime(*_, **__):
        raise RuntimeError("stable_baselines3 is required for SAC training")

    monkeypatch.setattr(trainer, "train_sac", raise_runtime)

    with pytest.raises(SystemExit) as excinfo:
        trainer.run_training(args)
    assert "stable_baselines3 is required" in str(excinfo.value)
