from __future__ import annotations

from datetime import datetime, timezone

from research.envs.signal_weight_env import SignalWeightEnvConfig, SignalWeightTradingEnv
from research.eval.evaluate import EvaluationMetadata, MetricConfig, evaluation_payload, from_rollout
from research.risk import RiskConfig
from research.runners.rollout import run_rollout

UTC = timezone.utc


class ConcentratedPolicy:
    def decide(self, features):
        return 1.0 if float(features.get("bias", 0.0)) > 0.5 else 0.0


def _build_rows():
    timestamps = [
        datetime(2023, 1, 2 + idx, 16, tzinfo=UTC) for idx in range(4)
    ]
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.append(
            {
                "timestamp": ts,
                "panel": {
                    "AAA": {"close": 100.0 + idx, "bias": 1.0},
                    "BBB": {"close": 50.0 + idx, "bias": 0.0},
                },
            }
        )
    return rows


def test_rollout_projection_enforces_constraints():
    rows = _build_rows()
    risk_config = RiskConfig(max_weight=0.55, exposure_cap=0.7, min_cash=0.15, max_turnover_1d=0.4)
    env_config = SignalWeightEnvConfig(transaction_cost_bp=0.0, risk_config=risk_config)
    env = SignalWeightTradingEnv(rows, config=env_config, observation_columns=("close", "bias"))
    policy = ConcentratedPolicy()
    inputs_used = {"fixture": "sha256:dummied"}
    result = run_rollout(env, policy, inputs_used=inputs_used, metadata={"risk_config": risk_config.to_dict()})

    effective_cap = min(risk_config.exposure_cap or 1.0, 1.0 - (risk_config.min_cash or 0.0))
    for entry in result.weights:
        assert all(value >= -1e-12 for value in entry.values())
        assert all(value <= risk_config.max_weight + 1e-12 for value in entry.values())
        total = sum(entry.values())
        assert total <= effective_cap + 1e-12

    series = from_rollout(
        timestamps=result.timestamps,
        account_values=result.account_values,
        weights=result.weights,
        transaction_costs=result.transaction_costs,
        symbols=result.symbols,
        rollout_metadata=result.metadata,
    )
    metadata = EvaluationMetadata(
        symbols=result.symbols,
        start_date="2023-01-02",
        end_date="2023-01-05",
        interval="daily",
        feature_set="bias_only",
        policy_id="concentrated",
        run_id="constraint_test",
    )
    payload_a = evaluation_payload(series, metadata, config=MetricConfig(risk_config=risk_config))

    risk_safety = payload_a["safety"]
    assert risk_safety["constraint_violations_count"] == 0.0
    assert risk_safety["max_weight_violation_count"] == 0.0
    assert risk_safety["exposure_violation_count"] == 0.0
    assert risk_safety["turnover_violation_count"] == 0.0
    assert payload_a["trading"]["avg_cash"] >= risk_config.min_cash

    # Determinism: rerun rollout and metrics
    env2 = SignalWeightTradingEnv(rows, config=env_config, observation_columns=("close", "bias"))
    result_b = run_rollout(env2, policy, inputs_used=inputs_used, metadata={"risk_config": risk_config.to_dict()})
    series_b = from_rollout(
        timestamps=result_b.timestamps,
        account_values=result_b.account_values,
        weights=result_b.weights,
        transaction_costs=result_b.transaction_costs,
        symbols=result_b.symbols,
        rollout_metadata=result_b.metadata,
    )
    payload_b = evaluation_payload(series_b, metadata, config=MetricConfig(risk_config=risk_config))
    assert payload_a == payload_b
