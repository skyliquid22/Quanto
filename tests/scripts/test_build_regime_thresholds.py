from __future__ import annotations

import pytest

from scripts.build_regime_thresholds import (
    compute_bucket_distribution,
    validate_bucket_distribution,
)


def test_bucket_distribution_sums_to_one() -> None:
    vol = [0.02, 0.02, 0.005, 0.005]
    trend = [0.01, -0.01, 0.0, 0.003]
    distribution = compute_bucket_distribution(
        vol,
        trend,
        vol_high=0.012,
        deadzone=0.002,
    )
    total = sum(distribution.values())
    assert abs(total - 1.0) < 1e-6


def test_bucket_distribution_min_share() -> None:
    distribution = {
        "high_vol_trend_up": 0.01,
        "high_vol_trend_down": 0.02,
        "high_vol_flat": 0.02,
        "low_vol_trend_up": 0.25,
        "low_vol_trend_down": 0.30,
        "low_vol_flat": 0.40,
    }
    with pytest.raises(SystemExit):
        validate_bucket_distribution(distribution, min_share=0.02)
