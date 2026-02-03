"""Fundamentals feature builder with point-in-time alignment."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

try:  # pragma: no cover - pandas optional
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

from research.datasets.canonical_fundamentals_loader import load_canonical_fundamentals

if TYPE_CHECKING:  # pragma: no cover
    from research.features.feature_registry import FeatureBuildContext


FUNDAMENTALS_V1_COLUMNS: Tuple[str, ...] = (
    "fund_ttm_revenue",
    "fund_ttm_net_income",
    "fund_ttm_operating_income",
    "fund_ttm_free_cash_flow",
    "fund_ttm_eps",
    "fund_total_assets",
    "fund_total_liabilities",
    "fund_shareholder_equity",
    "fund_shares_outstanding",
    "fund_row_valid",
    "fund_stale",
)

_TTM_FIELDS = (
    "revenue",
    "net_income",
    "operating_income",
    "free_cash_flow",
    "eps",
)
_BALANCE_FIELDS = (
    "total_assets",
    "total_liabilities",
    "shareholder_equity",
    "shares_outstanding",
)
_ANNUAL_LAG_DAYS = 90
_QUARTER_LAG_DAYS = 45
_STALE_DAYS = 180
_LOOKBACK_DAYS = 400


def attach_fundamentals_features(
    equity_frame: "pd.DataFrame",
    context: FeatureBuildContext,
) -> "pd.DataFrame":
    """Attach fundamentals to a per-symbol equity feature frame."""

    _ensure_pandas_available()
    if context is None:
        raise ValueError("Feature build context is required for fundamentals_v1")
    slices, hashes = load_canonical_fundamentals(
        [context.symbol],
        context.start_date,
        context.end_date,
        data_root=context.data_root,
        lookback_days=_LOOKBACK_DAYS,
    )
    context.inputs_used.update(hashes)
    slice_data = slices.get(context.symbol)
    fundamentals = slice_data.frame.copy() if slice_data else pd.DataFrame()
    return _attach_fundamentals(equity_frame, fundamentals)


def _attach_fundamentals(equity_frame: "pd.DataFrame", fundamentals: "pd.DataFrame") -> "pd.DataFrame":
    equity = equity_frame.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    equity.sort_values("timestamp", inplace=True, kind="mergesort")
    equity.reset_index(drop=True, inplace=True)

    if fundamentals.empty:
        return _assign_empty_fundamentals(equity)

    fundamentals = _prepare_fundamentals(fundamentals)
    if fundamentals.empty:
        return _assign_empty_fundamentals(equity)

    daily = _build_daily_fundamentals(fundamentals)
    if daily.empty or daily["as_of_date"].isna().all():
        return _assign_empty_fundamentals(equity)

    daily.sort_values("as_of_date", inplace=True, kind="mergesort")
    equity_sorted = equity.sort_values("timestamp", kind="mergesort")
    merged = pd.merge_asof(
        equity_sorted,
        daily,
        left_on="timestamp",
        right_on="as_of_date",
        direction="backward",
    )

    age_days = (merged["timestamp"].dt.normalize() - merged["as_of_date"].dt.normalize()).dt.days
    stale_mask = age_days > _STALE_DAYS
    has_asof = merged["as_of_date"].notna()
    for column in FUNDAMENTALS_V1_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    for column in FUNDAMENTALS_V1_COLUMNS:
        if column.startswith("fund_") and column not in {"fund_row_valid", "fund_stale"}:
            merged.loc[stale_mask, column] = np.nan
    merged["fund_stale"] = stale_mask.fillna(False)
    merged["fund_row_valid"] = (has_asof & ~merged["fund_stale"]).fillna(False)
    merged["fund_row_valid"] = merged["fund_row_valid"].astype(bool)
    merged["fund_stale"] = merged["fund_stale"].astype(bool)
    merged.drop(columns=["as_of_date"], inplace=True, errors="ignore")
    return merged


def _prepare_fundamentals(frame: "pd.DataFrame") -> "pd.DataFrame":
    data = frame.copy()
    data["report_date"] = pd.to_datetime(data["report_date"], utc=True, errors="coerce").dt.normalize()
    data["filing_date"] = pd.to_datetime(data.get("filing_date"), utc=True, errors="coerce").dt.normalize()
    data = data.dropna(subset=["report_date"])
    data["period"] = data.apply(
        lambda row: _normalize_period(row.get("period"), row.get("fiscal_period")),
        axis=1,
    )
    for column in _TTM_FIELDS + _BALANCE_FIELDS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = np.nan
    data["as_of_date"] = data["filing_date"]
    lag_days = data["period"].map({"annual": _ANNUAL_LAG_DAYS, "quarterly": _QUARTER_LAG_DAYS})
    fallback = data["report_date"] + pd.to_timedelta(lag_days.fillna(_QUARTER_LAG_DAYS), unit="D")
    data["as_of_date"] = data["as_of_date"].fillna(fallback)
    return data


def _build_daily_fundamentals(frame: "pd.DataFrame") -> "pd.DataFrame":
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_values(["report_date", "filing_date"], kind="mergesort")
    quarterly = frame[frame["period"] == "quarterly"].copy()
    annual = frame[frame["period"] == "annual"].copy()

    report_dates = pd.Index(sorted(frame["report_date"].dropna().unique()))
    if report_dates.empty:
        return pd.DataFrame()

    ttm = _compute_ttm(quarterly, report_dates)
    annual_snapshot = _annual_fallback(annual, report_dates)
    ttm = ttm.where(ttm.notna(), annual_snapshot)

    period_rank = frame["period"].map({"annual": 0, "quarterly": 1}).fillna(0)
    frame = frame.assign(_period_rank=period_rank)
    latest_rows = frame.sort_values("_period_rank", kind="mergesort").groupby("report_date").tail(1)
    latest_rows = latest_rows.set_index("report_date")
    latest_rows = latest_rows.reindex(report_dates)

    output = pd.DataFrame(index=report_dates)
    output["as_of_date"] = latest_rows["as_of_date"]
    for field in _TTM_FIELDS:
        output[f"fund_ttm_{field}"] = ttm[field]
    output["fund_total_assets"] = latest_rows["total_assets"]
    output["fund_total_liabilities"] = latest_rows["total_liabilities"]
    output["fund_shareholder_equity"] = latest_rows["shareholder_equity"]
    output["fund_shares_outstanding"] = latest_rows["shares_outstanding"]

    output.reset_index(drop=True, inplace=True)
    return output


def _compute_ttm(quarterly: "pd.DataFrame", report_dates: "pd.Index") -> "pd.DataFrame":
    if quarterly.empty:
        return pd.DataFrame(index=report_dates, columns=_TTM_FIELDS, dtype="float64")
    quarterly = quarterly.set_index("report_date").sort_index()
    ttm = quarterly[list(_TTM_FIELDS)].rolling(4, min_periods=4).sum()
    return ttm.reindex(report_dates)


def _annual_fallback(annual: "pd.DataFrame", report_dates: "pd.Index") -> "pd.DataFrame":
    if annual.empty:
        return pd.DataFrame(index=report_dates, columns=_TTM_FIELDS, dtype="float64")
    annual = annual.set_index("report_date").sort_index()
    annual = annual[list(_TTM_FIELDS)]
    return annual.reindex(report_dates, method="ffill")


def _assign_empty_fundamentals(frame: "pd.DataFrame") -> "pd.DataFrame":
    for column in FUNDAMENTALS_V1_COLUMNS:
        if column in frame.columns:
            continue
        if column in {"fund_row_valid", "fund_stale"}:
            frame[column] = False
        else:
            frame[column] = np.nan
    frame["fund_row_valid"] = frame["fund_row_valid"].astype(bool)
    frame["fund_stale"] = frame["fund_stale"].astype(bool)
    return frame


def _normalize_period(period: object, fiscal_period: object) -> str:
    text = str(period or "").strip().lower()
    if text in {"quarter", "quarterly", "q", "qtr"}:
        return "quarterly"
    if text in {"annual", "fy", "year", "yearly"}:
        return "annual"
    fiscal = str(fiscal_period or "").strip().upper()
    if fiscal.startswith("Q"):
        return "quarterly"
    if fiscal.startswith("FY") or fiscal.startswith("A"):
        return "annual"
    return text


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for fundamentals_v1 feature builder") from _PANDAS_ERROR


__all__ = ["FUNDAMENTALS_V1_COLUMNS", "attach_fundamentals_features"]
