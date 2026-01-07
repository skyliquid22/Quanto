from __future__ import annotations

import math

from research.risk import RiskConfig, project_weights


def test_long_only_clips_negatives():
    cfg = RiskConfig(long_only=True)
    prev = [0.2, 0.3]
    result = project_weights([-0.2, 0.3], prev, cfg)
    assert result[0] == 0.0
    assert result[1] >= 0.0


def test_max_weight_cap_enforced():
    cfg = RiskConfig(max_weight=0.4)
    projected = project_weights([0.8, 0.2], [0.3, 0.3], cfg)
    assert projected[0] == 0.4
    assert math.isclose(sum(projected), 0.6)


def test_exposure_cap_scales_weights():
    cfg = RiskConfig(exposure_cap=0.7)
    projected = project_weights([0.6, 0.8], [0.2, 0.5], cfg)
    assert math.isclose(sum(projected), 0.7)
    ratio = projected[0] / projected[1]
    assert math.isclose(ratio, 0.6 / 0.8)


def test_min_cash_limits_total_exposure():
    cfg = RiskConfig(min_cash=0.25)
    projected = project_weights([1.0, 0.8], [0.1, 0.2], cfg)
    assert sum(projected) <= 0.75 + 1e-12


def test_turnover_cap_enforced_with_reprojection():
    cfg = RiskConfig(max_turnover_1d=0.3)
    prev = [0.4, 0.2]
    result = project_weights([1.0, 0.0], prev, cfg)
    turnover = abs(result[0] - prev[0]) + abs(result[1] - prev[1])
    assert turnover <= 0.3 + 1e-12
    assert all(value >= 0.0 for value in result)


def test_projection_determinism():
    cfg = RiskConfig(max_weight=0.5, exposure_cap=0.8, max_turnover_1d=0.4)
    prev = [0.3, 0.3]
    raw = [0.9, 0.1]
    first = project_weights(raw, prev, cfg)
    second = project_weights(raw, prev, cfg)
    assert first == second


def test_boundary_conditions_respected():
    cfg = RiskConfig()
    assert project_weights([0.0, 0.0], [0.0, 0.0], cfg) == [0.0, 0.0]
    eps = project_weights([1e-14, 1e-14], [0.0, 0.0], cfg)
    assert eps == [0.0, 0.0]
    feasible = project_weights([0.25, 0.25], [0.25, 0.25], cfg)
    assert feasible == [0.25, 0.25]
