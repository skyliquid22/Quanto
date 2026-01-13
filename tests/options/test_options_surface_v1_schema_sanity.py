from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from infra.normalization.options_surface import NORMALIZED_OPTIONS_SURFACE_COLUMNS, OPT_COVERAGE_COLUMNS, normalize_ivol_surface


def test_options_surface_schema_and_coverage_flags():
    fixture = json.loads(Path("tests/fixtures/options_surface_sample.json").read_text(encoding="utf-8"))
    rows = normalize_ivol_surface(fixture, allowed_symbols=("AAA", "BBB"), start_date="2024-01-02", end_date="2024-01-03")
    frame = pd.DataFrame(rows)
    assert set(NORMALIZED_OPTIONS_SURFACE_COLUMNS).issubset(frame.columns)

    aaa_latest = frame[(frame["symbol"] == "AAA") & (frame["date"] == "2024-01-02")].iloc[0]
    assert aaa_latest["OPT:OI:TOTAL"] == 1600.0
    assert aaa_latest["OPT:IVX:30"] == 0.235
    assert aaa_latest["OPT:IVX:TERM_SLOPE_30_90"] == pytest.approx(0.063)
    assert bool(aaa_latest["OPT:COVERAGE:HAS_IVX"]) is True
    assert bool(aaa_latest["OPT:COVERAGE:ROW_VALID"]) is True

    aaa_missing = frame[(frame["symbol"] == "AAA") & (frame["date"] == "2024-01-03")].iloc[0]
    assert pd.isna(aaa_missing["OPT:IVR:30"])
    assert bool(aaa_missing["OPT:COVERAGE:HAS_IVR"]) is False

    coverage_flags = frame.loc[:, list(OPT_COVERAGE_COLUMNS)]
    assert all(dtype == bool or dtype == "bool" for dtype in coverage_flags.dtypes)
