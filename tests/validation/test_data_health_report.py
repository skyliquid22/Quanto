from __future__ import annotations

from datetime import date

import pandas as pd

from research.validation.data_health import (
    compute_canonical_health,
    compute_feature_health,
    compute_fundamentals_health,
    evaluate_thresholds,
)
from research.datasets.canonical_equity_loader import CanonicalEquitySlice


def _make_slice(symbol: str, dates: list[str]) -> CanonicalEquitySlice:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates, utc=True),
            "symbol": symbol,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    )
    frame.set_index("timestamp", inplace=True)
    return CanonicalEquitySlice(symbol=symbol, frame=frame, file_paths=[])


def test_canonical_health_union_and_intersection() -> None:
    slice_a = _make_slice("AAA", ["2025-01-02", "2025-01-03", "2025-01-06"])
    slice_b = _make_slice("BBB", ["2025-01-02", "2025-01-06"])
    slices = {"AAA": slice_a, "BBB": slice_b}
    report_union = compute_canonical_health(
        slices,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 6),
        calendar_mode="union",
    )
    b_union = report_union["summary_by_symbol"]["BBB"]
    assert b_union["missing_count"] == 1
    assert b_union["missing_ranges"][0]["start"] == "2025-01-03"
    assert b_union["missing_ranges"][0]["end"] == "2025-01-03"

    report_intersection = compute_canonical_health(
        slices,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 6),
        calendar_mode="intersection",
    )
    b_intersection = report_intersection["summary_by_symbol"]["BBB"]
    assert b_intersection["missing_count"] == 0
    assert b_intersection["missing_ranges"] == []


def test_feature_health_nan_ratios_and_thresholds() -> None:
    frame_a = pd.DataFrame({"x": [1.0, None], "y": [1.0, 2.0]})
    frame_b = pd.DataFrame({"x": [1.0, 2.0], "y": [None, None]})
    report = compute_feature_health({"AAA": frame_a, "BBB": frame_b}, ["x", "y"])
    by_column = report["summary_by_column"]
    assert by_column["x"]["nan_ratio"] == 0.25
    assert by_column["y"]["nan_ratio"] == 0.5

    failures = evaluate_thresholds(
        canonical_report={"overall": {"missing_ratio": 0.1}},
        feature_report=report,
        max_missing_ratio=0.05,
        max_nan_ratio=0.4,
    )
    assert "missing_ratio_exceeded" in failures
    assert "nan_ratio_exceeded" in failures


def test_fundamentals_health_staleness() -> None:
    frame_a = pd.DataFrame(
        {
            "report_date": ["2024-12-31"],
            "filing_date": ["2025-02-15"],
            "period": ["quarterly"],
            "fiscal_period": ["Q4"],
        }
    )
    frame_b = pd.DataFrame(
        {
            "report_date": ["2025-09-30"],
            "filing_date": ["2025-11-05"],
            "period": ["quarterly"],
            "fiscal_period": ["Q3"],
        }
    )
    report = compute_fundamentals_health(
        {"AAA": frame_a, "BBB": frame_b},
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        stale_days=180,
    )
    summary = report["summary_by_symbol"]
    assert summary["AAA"]["stale_at_end"] is True
    assert summary["BBB"]["stale_at_end"] is False
