from __future__ import annotations

import numpy as np

from research.hierarchy.allocator_registry import build_allocator


def test_defensive_cash_allocator_deterministic_zero_exposure():
    allocator = build_allocator({"type": "defensive_cash"}, num_assets=3)
    obs = np.ones(5, dtype=np.float32)
    context = {"num_assets": 3}

    first = allocator.act(obs, context=context)
    second = allocator.act(obs, context=context)

    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.allclose(first, np.zeros(3, dtype=np.float32))


def test_defensive_cash_allocator_target_exposure_uniform():
    allocator = build_allocator({"type": "defensive_cash", "target_exposure": 0.3}, num_assets=4)
    obs = np.zeros(2, dtype=np.float32)
    context = {"num_assets": 4}

    weights = allocator.act(obs, context=context)

    assert weights.dtype == np.float32
    assert weights.shape == (4,)
    assert np.isclose(weights.sum(), 0.3, atol=1e-9)
    assert np.allclose(weights, np.full(4, 0.075, dtype=np.float32))
