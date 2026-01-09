from __future__ import annotations

import pandas as pd
import pytest

from research.features.label_eng import (
    compute_forward_drawdown_labels,
    compute_forward_return_labels,
    compute_regime_transition_flags,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="D"),
            "close": [100.0, 101.0, 98.0, 99.0, 105.0],
            "regime_state": ["low", "low", "high", "high", "low"],
        }
    )


def test_forward_return_labels_no_lookahead():
    frame = _sample_frame()
    labeled = compute_forward_return_labels(frame, horizons=(1, 2))
    assert labeled.loc[0, "label_fwd_return_1"] == pytest.approx(0.01)
    assert labeled.loc[0, "label_fwd_return_2"] == pytest.approx(-0.02)
    assert labeled.loc[4, "label_fwd_return_1"] == 0.0


def test_forward_drawdown_labels_detects_min_path():
    frame = compute_forward_return_labels(_sample_frame(), horizons=(1,))
    labeled = compute_forward_drawdown_labels(frame, horizons=(2,))
    assert labeled.loc[0, "label_fwd_drawdown_2"] == pytest.approx(-0.02)
    assert labeled.loc[3, "label_fwd_drawdown_2"] == pytest.approx(0.0)


def test_regime_transition_flags():
    frame = _sample_frame()
    labeled = compute_regime_transition_flags(frame, horizons=(1, 2))
    assert labeled.loc[0, "label_regime_transition_1"] == 0
    assert labeled.loc[1, "label_regime_transition_1"] == 1
    assert labeled.loc[3, "label_regime_transition_2"] == 0
