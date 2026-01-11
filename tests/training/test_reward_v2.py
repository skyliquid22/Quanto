from __future__ import annotations

import math

import pytest

from research.training.rewards.reward_v2 import StableRewardV2


def test_turnover_penalty_increases_with_weight_change():
    reward = StableRewardV2(turnover_scale=0.5, drawdown_scale=0.0)
    reward.reset()
    base = math.log(1.01)
    shaped, components = reward.compute(base_reward=base, info={"weight_realized": {"AAA": 0.1}}, step_index=0)
    assert shaped == pytest.approx(base)
    shaped2, components2 = reward.compute(base_reward=base, info={"weight_realized": {"AAA": 0.9}}, step_index=1)
    assert components2["turnover"] > 0.0
    assert shaped2 < base


def test_drawdown_penalty_only_on_increase():
    reward = StableRewardV2(drawdown_scale=1.0, turnover_scale=0.0)
    reward.reset()
    base = math.log(1.0)
    reward.compute(base_reward=base, info={"portfolio_value": 105.0}, step_index=0)
    shaped, components = reward.compute(base_reward=math.log(99.0 / 105.0), info={"portfolio_value": 99.0}, step_index=1)
    assert components["drawdown"] > 0.0
    assert components["drawdown_increase"] == components["drawdown"]
    shaped2, components2 = reward.compute(base_reward=math.log(98.5 / 99.0), info={"portfolio_value": 98.5}, step_index=2)
    assert components2["drawdown_increase"] >= 0.0


def test_reset_clears_state():
    reward = StableRewardV2(turnover_scale=1.0)
    reward.compute(base_reward=0.0, info={"weight_realized": {"AAA": 1.0}}, step_index=0)
    reward.reset()
    shaped, components = reward.compute(base_reward=0.0, info={"weight_realized": {"AAA": 1.0}}, step_index=1)
    assert components["turnover"] == 0.0


def test_missing_keys_degrades_to_base_reward():
    reward = StableRewardV2(turnover_scale=1.0, drawdown_scale=1.0)
    shaped, components = reward.compute(base_reward=0.05, info={}, step_index=0)
    assert shaped == pytest.approx(0.05)
    assert components["turnover"] == 0.0
    assert components["drawdown_penalty"] == 0.0
