"""Reward version modules registering with :mod:`research.training.reward_registry`."""

# Importing the modules registers their RewardSpec definitions as a side-effect.
from research.training.rewards import reward_v1 as _reward_v1  # noqa: F401
from research.training.rewards import reward_v2 as _reward_v2  # noqa: F401

__all__ = ["_reward_v1", "_reward_v2"]
