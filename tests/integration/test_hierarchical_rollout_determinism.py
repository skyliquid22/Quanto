from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.eval.evaluate import EvaluationMetadata, evaluation_payload, from_rollout
from research.eval.metrics import MetricConfig
from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS
from research.hierarchy.allocator_registry import build_allocator
from research.hierarchy.controller import ControllerConfig, ModeController
from research.hierarchy.policy_wrapper import HierarchicalPolicy, run_hierarchical_rollout
from research.regime import RegimeState
from research.risk import RiskConfig


UTC = timezone.utc
OBSERVATION_COLUMNS = ("close", "sma_fast", "sma_slow", "sma_diff", "sma_signal") + REGIME_FEATURE_COLUMNS


def _build_rows() -> list[dict]:
    base = datetime(2023, 1, 2, 16, tzinfo=UTC)
    closes = {
        "AAPL": [150.0, 151.0, 152.0, 153.0, 154.0, 155.0],
        "MSFT": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0],
    }
    regimes = [
        (0.7, 0.0, 0.0, 0.0),
        (0.1, 0.4, 0.3, 0.0),
        (0.1, 0.5, 0.4, 0.0),
        (0.6, 0.0, 0.0, 0.0),
        (0.1, 0.5, 0.4, 0.0),
        (0.1, 0.5, 0.4, 0.0),
    ]
    rows: list[dict] = []
    for idx, regime in enumerate(regimes):
        timestamp = base + timedelta(days=idx)
        panel = {}
        for symbol, series in closes.items():
            close = series[idx]
            sma_fast = close + (1.0 if idx % 2 == 0 else -1.0)
            sma_slow = close
            panel[symbol] = {
                "close": close,
                "sma_fast": sma_fast,
                "sma_slow": sma_slow,
                "sma_diff": sma_fast - sma_slow,
                "sma_signal": 1.0,
            }
        state = RegimeState(features=np.asarray(regime, dtype="float64"), feature_names=REGIME_FEATURE_COLUMNS)
        rows.append({"timestamp": timestamp, "panel": panel, "regime_state": state})
    return rows


def _run_once(rows):
    config = SignalWeightEnvConfig(transaction_cost_bp=0.0, risk_config=RiskConfig())
    env = SignalWeightTradingEnv(rows, config=config, observation_columns=OBSERVATION_COLUMNS)
    controller_cfg = ControllerConfig(
        update_frequency="every_k_steps",
        min_hold_steps=0,
        vol_threshold_high=0.5,
        trend_threshold_high=0.3,
        dispersion_threshold_high=0.2,
        fallback_mode="neutral",
        every_k_steps=1,
    )
    controller = ModeController(config=controller_cfg)
    allocator_cfg = {
        "neutral": {"type": "equal_weight"},
        "risk_on": {"type": "sma", "mode": "hard"},
        "defensive": {"type": "equal_weight"},
    }
    allocators = {mode: build_allocator(cfg, num_assets=env.num_assets) for mode, cfg in allocator_cfg.items()}
    policy = HierarchicalPolicy(controller, allocators, config.risk_config)
    rollout = run_hierarchical_rollout(env, policy, rows=rows)
    metadata = EvaluationMetadata(
        symbols=tuple(env.symbols),
        start_date=rows[0]["timestamp"].date().isoformat(),
        end_date=rows[-1]["timestamp"].date().isoformat(),
        interval="daily",
        feature_set="test_features",
        policy_id="hierarchical_test",
        run_id="test_run",
        policy_details={"type": "hierarchical"},
    )
    series = from_rollout(
        timestamps=rollout.timestamps,
        account_values=rollout.account_values,
        weights=rollout.weights,
        transaction_costs=rollout.transaction_costs,
        symbols=rollout.symbols,
        rollout_metadata={"modes": rollout.modes},
        regime_features=rollout.regime_features,
        regime_feature_names=rollout.regime_feature_names,
        modes=rollout.modes,
    )
    payload = evaluation_payload(series, metadata, config=MetricConfig(risk_config=config.risk_config))
    return rollout, payload


def test_hierarchical_rollout_determinism():
    rows = _build_rows()
    first, payload_first = _run_once(rows)
    second, payload_second = _run_once(rows)

    assert first.account_values == second.account_values
    assert first.modes == second.modes
    assert payload_first == payload_second
    assert {"risk_on", "defensive"}.issubset(set(first.modes))
