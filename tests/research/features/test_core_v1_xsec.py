from __future__ import annotations

import json
from datetime import timezone
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from research.datasets.canonical_equity_loader import CanonicalEquitySlice
from research.features.feature_eng import build_universe_feature_results
from research.features.feature_registry import CORE_V1_XSEC_OBSERVATION_COLUMNS, build_universe_feature_panel

UTC = timezone.utc


def _build_slices(periods: int = 90) -> Tuple[Tuple[str, ...], Dict[str, CanonicalEquitySlice], pd.DatetimeIndex]:
    symbols = ("AAA", "BBB", "CCC")
    start = pd.Timestamp("2023-01-03", tz=UTC)
    dates = pd.date_range(start, periods=periods, freq="B", tz=UTC)
    slices: Dict[str, CanonicalEquitySlice] = {}
    for idx, symbol in enumerate(symbols):
        base = 100.0 + idx * 5.0
        close = base + np.linspace(0.0, 10.0, len(dates))
        volume = 1_000_000 + idx * 5_000 + np.linspace(0.0, 1_000.0, len(dates))
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "open": close - 0.4,
                "high": close + 0.6,
                "low": close - 0.8,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        frame.index.name = "timestamp"
        slices[symbol] = CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=[])
    return symbols, slices, dates


def _build_panel(symbols: Sequence[str], slices: Dict[str, CanonicalEquitySlice], dates: pd.DatetimeIndex):
    start_date = dates[0].date()
    end_date = dates[-1].date()
    feature_results = build_universe_feature_results(
        "core_v1_xsec",
        slices,
        symbol_order=symbols,
        start_date=start_date,
        end_date=end_date,
        data_root=None,
    )
    panel = build_universe_feature_panel(
        feature_results,
        symbol_order=symbols,
        calendar=dates,
        forward_fill_limit=3,
        regime_feature_set=None,
    )
    return panel


def test_core_v1_xsec_contract_and_no_regime_state():
    symbols, slices, dates = _build_slices()
    panel = _build_panel(symbols, slices, dates)

    for row in panel.rows:
        assert "regime_state" not in row
        for symbol in symbols:
            features = row["panel"][symbol]
            assert tuple(features.keys()) == CORE_V1_XSEC_OBSERVATION_COLUMNS


def test_core_v1_xsec_no_lookahead_on_tail_mutation():
    symbols, slices, dates = _build_slices()
    base_panel = _build_panel(symbols, slices, dates)

    _, mutated_slices, _ = _build_slices()
    target_frame = mutated_slices["BBB"].frame
    last_idx = target_frame.index[-1]
    target_frame.loc[last_idx, "close"] += 5.0
    mutated_panel = _build_panel(symbols, mutated_slices, dates)

    assert len(base_panel.rows) == len(mutated_panel.rows)
    for idx in range(len(base_panel.rows) - 1):
        assert base_panel.rows[idx] == mutated_panel.rows[idx]


def test_core_v1_xsec_deterministic_rows():
    symbols, slices, dates = _build_slices()
    panel_a = _build_panel(symbols, slices, dates)
    # rebuild from fresh slices to ensure determinism
    _, slices_b, dates_b = _build_slices()
    panel_b = _build_panel(symbols, slices_b, dates_b)

    def _canonicalize(rows):
        def _default(value):
            if hasattr(value, "isoformat"):
                return value.isoformat()
            return value

        return json.loads(json.dumps(rows, default=_default, sort_keys=True))

    assert _canonicalize(panel_a.rows) == _canonicalize(panel_b.rows)


def test_core_v1_xsec_cross_sectional_variation():
    symbols, slices, dates = _build_slices()
    panel = _build_panel(symbols, slices, dates)

    differing_found = False
    for row in panel.rows[10:]:  # skip warmup to allow dispersion
        values = {symbol: row["panel"][symbol]["ret_1d_rank"] for symbol in symbols}
        if len({round(value, 6) for value in values.values()}) > 1:
            differing_found = True
            break
    assert differing_found, "Expected at least one timestamp with differing cross-sectional ranks"
