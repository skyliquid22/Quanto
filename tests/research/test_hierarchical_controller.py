from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from research.hierarchy.controller import ControllerConfig, ModeController
from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS


UTC = timezone.utc


def _feature_matrix(patterns: list[tuple[float, float, float, float]]) -> np.ndarray:
    return np.asarray(patterns, dtype=float)


def test_mode_controller_weekly_schedule_and_hold():
    config = ControllerConfig(
        update_frequency="weekly",
        min_hold_steps=2,
        vol_threshold_high=0.5,
        trend_threshold_high=0.2,
        dispersion_threshold_high=0.1,
        fallback_mode="neutral",
    )
    controller = ModeController(config=config)
    dates = np.asarray(
        [
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 10, tzinfo=UTC),
            datetime(2023, 1, 11, tzinfo=UTC),
            datetime(2023, 1, 18, tzinfo=UTC),
            datetime(2023, 1, 19, tzinfo=UTC),
        ],
        dtype=object,
    )
    # Columns follow REGIME_FEATURE_COLUMNS order.
    features = _feature_matrix(
        [
            (0.6, 0.0, 0.0, 0.0),  # defensive trigger
            (0.1, 0.3, 0.2, 0.0),  # risk_on but hold prevents change
            (0.1, 0.4, 0.4, 0.0),  # new week + hold satisfied -> risk_on
            (0.3, -0.1, 0.05, 0.0),  # same week -> remains risk_on
            (0.7, 0.0, 0.0, 0.0),  # new week -> defensive
            (0.1, 0.4, 0.3, 0.0),  # hold prevents immediate switch
        ]
    )
    expected = ["defensive", "defensive", "risk_on", "risk_on", "defensive", "defensive"]
    outputs: list[str] = []
    prev_mode: str | None = None
    for t in range(len(expected)):
        mode = controller.select_mode(t, dates, features, prev_mode)
        outputs.append(mode)
        prev_mode = mode
    assert outputs == expected

    controller.reset()
    prev_mode = None
    replay: list[str] = []
    for t in range(len(expected)):
        mode = controller.select_mode(t, dates, features, prev_mode)
        replay.append(mode)
        prev_mode = mode
    assert replay == expected
