import numpy as np
import pandas as pd

from research.features.regime_features_v1 import REGIME_FEATURE_COLUMNS, compute_regime_features


def _manual_regime_features(close_panel: pd.DataFrame, window: int) -> pd.DataFrame:
    returns = close_panel.pct_change().fillna(0.0)
    rows = []
    for idx in range(len(close_panel.index)):
        start = max(0, idx - window + 1)
        window_slice = returns.iloc[start : idx + 1]
        market_returns = window_slice.mean(axis=1)
        if len(market_returns) >= 2:
            market_vol = market_returns.std(ddof=0)
        else:
            market_vol = 0.0
        market_trend = market_returns.mean() if not market_returns.empty else 0.0
        dispersion_daily = window_slice.std(axis=1, ddof=0)
        dispersion = dispersion_daily.mean() if not dispersion_daily.empty else 0.0
        corr_mean = 0.0
        if window_slice.shape[1] >= 2 and len(window_slice) >= 2:
            corr = window_slice.corr()
            if corr.shape[0] >= 2:
                matrix = corr.to_numpy()
                upper = np.triu_indices_from(matrix, k=1)
                upper_values = matrix[upper]
                finite = upper_values[np.isfinite(upper_values)]
                corr_mean = float(finite.mean()) if finite.size else 0.0
        rows.append(
            (
                float(market_vol),
                float(market_trend),
                float(dispersion),
                float(corr_mean),
            )
        )
    manual = pd.DataFrame(rows, index=close_panel.index, columns=REGIME_FEATURE_COLUMNS)
    return manual.fillna(0.0)


def test_regime_features_match_manual_computation():
    index = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
    panel = pd.DataFrame(
        {
            "AAA": [100.0, 102.0, 101.0, 103.0, 104.0, 105.0],
            "BBB": [200.0, 201.0, 199.0, 202.0, 204.0, 207.0],
            "CCC": [300.0, 302.0, 301.0, 304.0, 303.0, 305.0],
        },
        index=index,
    )
    window = 3
    expected = _manual_regime_features(panel, window)
    computed = compute_regime_features(panel, window=window)
    assert tuple(computed.columns) == REGIME_FEATURE_COLUMNS
    assert not computed.isna().any().any()
    pd.testing.assert_frame_equal(computed, expected)
