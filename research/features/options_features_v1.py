"""Options-derived daily feature engineering primitives."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

try:  # pragma: no cover - pandas optional in some environments
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

EPS = 1e-9
FORWARD_FILL_DAYS = 2
OPTION_FEATURE_COLUMNS = (
    "oi_total",
    "oi_call_total",
    "oi_put_total",
    "oi_put_call_ratio",
    "oi_change_1d",
    "oi_change_pct_1d",
    "oi_concentration_top10",
    "options_volume_total",
    "options_volume_call_total",
    "options_volume_put_total",
)
_LEVEL_COLUMNS = ("oi_total", "oi_call_total", "oi_put_total", "oi_concentration_top10")
_OI_COLUMNS = (
    "oi_total",
    "oi_call_total",
    "oi_put_total",
    "oi_put_call_ratio",
    "oi_change_1d",
    "oi_change_pct_1d",
    "oi_concentration_top10",
)
_VOLUME_COLUMNS = ("options_volume_total", "options_volume_call_total", "options_volume_put_total")


def compute_options_features(
    equity_df: "pd.DataFrame",
    reference_df: "pd.DataFrame",
    open_interest_df: "pd.DataFrame",
    ohlcv_df: "pd.DataFrame | None" = None,
    *,
    forward_fill_days: int = FORWARD_FILL_DAYS,
) -> "pd.DataFrame":
    """Aggregate per-contract canonical options data into daily underlying-level features."""

    _ensure_pandas_available()
    if "timestamp" not in equity_df or "close" not in equity_df:
        raise ValueError("equity_df must contain timestamp and close columns")

    eq = equity_df.copy()
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
    eq.sort_values("timestamp", inplace=True, kind="mergesort")
    eq.reset_index(drop=True, inplace=True)
    eq["_date"] = eq["timestamp"].dt.normalize()

    target_index = pd.Index(pd.unique(eq["_date"]), name="timestamp")
    oi_daily = _aggregate_open_interest(reference_df, open_interest_df)
    volume_daily = _aggregate_volume(reference_df, ohlcv_df)
    if not volume_daily.empty:
        collision = [col for col in volume_daily.columns if col in oi_daily.columns]
        if collision:
            volume_daily = volume_daily.drop(columns=collision)
    combined = oi_daily.join(volume_daily, how="outer")
    for column in OPTION_FEATURE_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.Series(dtype="float64")
    combined = combined.reindex(target_index)
    combined.index.name = "timestamp"
    combined = _apply_fill_policy(combined, forward_fill_days)
    for column in _VOLUME_COLUMNS:
        if column in combined:
            combined[column] = combined[column].fillna(0.0)
        else:
            combined[column] = 0.0
    combined = combined[[column for column in OPTION_FEATURE_COLUMNS]]
    combined = combined.reset_index().rename(columns={"timestamp": "_date"})

    merged = eq.merge(combined, on="_date", how="left")
    merged.drop(columns=["_date"], inplace=True)
    return merged


def _aggregate_open_interest(reference_df: "pd.DataFrame", open_interest_df: "pd.DataFrame") -> "pd.DataFrame":
    if open_interest_df is None or open_interest_df.empty:
        return pd.DataFrame(columns=_OI_COLUMNS).set_index(pd.Index([], name="timestamp"))

    reference = reference_df.copy() if reference_df is not None else pd.DataFrame(columns=["option_symbol", "option_type"])
    if "option_symbol" not in reference:
        reference["option_symbol"] = []
    if "option_type" not in reference:
        reference["option_type"] = []

    ref_map: Dict[str, str] = (
        reference.assign(option_symbol=reference["option_symbol"].astype(str).str.upper())
        .set_index("option_symbol")["option_type"]
        .to_dict()
    )

    oi = open_interest_df.copy()
    oi["timestamp"] = pd.to_datetime(oi["timestamp"], utc=True)
    oi["option_symbol"] = oi["option_symbol"].astype(str).str.upper()
    oi["_date"] = oi["timestamp"].dt.normalize()
    oi["option_type"] = oi["option_symbol"].map(ref_map)
    missing_type = oi["option_type"].isna()
    if missing_type.any():
        oi.loc[missing_type, "option_type"] = oi.loc[missing_type, "option_symbol"].map(_infer_option_type)
    oi = oi.dropna(subset=["option_type"])
    if oi.empty:
        return pd.DataFrame(columns=_OI_COLUMNS).set_index(pd.Index([], name="timestamp"))

    totals = oi.groupby("_date", sort=True)["open_interest"].sum()
    call_totals = oi.loc[oi["option_type"] == "call"].groupby("_date")["open_interest"].sum()
    put_totals = oi.loc[oi["option_type"] == "put"].groupby("_date")["open_interest"].sum()

    concentration = pd.Series(
        {key: _concentration_ratio(group) for key, group in oi.groupby("_date", sort=True)},
        name="oi_concentration_top10",
    )

    daily = pd.DataFrame(
        {
            "oi_total": totals,
            "oi_call_total": call_totals,
            "oi_put_total": put_totals,
            "oi_concentration_top10": concentration,
        }
    )
    daily["oi_put_call_ratio"] = (daily["oi_put_total"] + EPS) / (daily["oi_call_total"] + EPS)
    return daily


def _aggregate_volume(reference_df: "pd.DataFrame", ohlcv_df: "pd.DataFrame | None") -> "pd.DataFrame":
    if ohlcv_df is None or ohlcv_df.empty:
        return pd.DataFrame(columns=_VOLUME_COLUMNS).set_index(pd.Index([], name="timestamp"))

    reference = reference_df.copy() if reference_df is not None else pd.DataFrame(columns=["option_symbol", "option_type"])
    if "option_symbol" not in reference:
        reference["option_symbol"] = []
    if "option_type" not in reference:
        reference["option_type"] = []

    ref_map: Dict[str, str] = (
        reference.assign(option_symbol=reference["option_symbol"].astype(str).str.upper())
        .set_index("option_symbol")["option_type"]
        .to_dict()
    )

    ohlcv = ohlcv_df.copy()
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
    ohlcv["option_symbol"] = ohlcv["option_symbol"].astype(str).str.upper()
    ohlcv["_date"] = ohlcv["timestamp"].dt.normalize()
    ohlcv["option_type"] = ohlcv["option_symbol"].map(ref_map)
    missing_type = ohlcv["option_type"].isna()
    if missing_type.any():
        ohlcv.loc[missing_type, "option_type"] = ohlcv.loc[missing_type, "option_symbol"].map(_infer_option_type)

    totals = ohlcv.groupby("_date", sort=True)["volume"].sum()
    call_totals = ohlcv.loc[ohlcv["option_type"] == "call"].groupby("_date")["volume"].sum()
    put_totals = ohlcv.loc[ohlcv["option_type"] == "put"].groupby("_date")["volume"].sum()
    return pd.DataFrame(
        {
            "options_volume_total": totals,
            "options_volume_call_total": call_totals,
            "options_volume_put_total": put_totals,
        }
    )


def _apply_fill_policy(frame: "pd.DataFrame", forward_fill_days: int) -> "pd.DataFrame":
    ordered = frame.sort_index()
    for column in _LEVEL_COLUMNS:
        ordered[column] = ordered[column].ffill(limit=forward_fill_days)
    ordered["oi_put_call_ratio"] = (ordered["oi_put_total"] + EPS) / (ordered["oi_call_total"] + EPS)
    ordered["oi_concentration_top10"] = ordered["oi_concentration_top10"].ffill(limit=forward_fill_days)
    ordered["oi_change_1d"] = ordered["oi_total"].diff()
    ordered["oi_change_pct_1d"] = ordered["oi_change_1d"] / (ordered["oi_total"].shift(1) + EPS)
    ordered["oi_change_1d"] = ordered["oi_change_1d"].fillna(0.0)
    ordered["oi_change_pct_1d"] = ordered["oi_change_pct_1d"].replace([float("inf"), float("-inf")], 0.0)
    ordered["oi_change_pct_1d"] = ordered["oi_change_pct_1d"].fillna(0.0)
    ordered["oi_put_call_ratio"] = ordered["oi_put_call_ratio"].fillna(1.0)
    ordered["oi_concentration_top10"] = ordered["oi_concentration_top10"].fillna(0.0)
    return ordered


def _concentration_ratio(group: "pd.DataFrame") -> float:
    sorted_group = group.sort_values(["open_interest", "option_symbol"], ascending=[False, True])
    total = sorted_group["open_interest"].sum()
    if total <= 0:
        return 0.0
    top_sum = sorted_group.head(10)["open_interest"].sum()
    return float(top_sum / total if total else 0.0)


def _infer_option_type(option_symbol: Any) -> str | None:
    text = str(option_symbol or "").strip().upper()
    if len(text) < 15:
        return None
    marker = text[14]
    if marker == "C":
        return "call"
    if marker == "P":
        return "put"
    return None


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for options feature computation") from _PANDAS_ERROR


__all__ = ["OPTION_FEATURE_COLUMNS", "compute_options_features"]
