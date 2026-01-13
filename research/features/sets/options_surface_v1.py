"""Dense, availability-aware options surface feature builder."""

from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

try:  # pragma: no cover - pandas optional
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

from research.datasets.options_surface_loader import OptionsSurfaceSlice, load_options_surface
from research.features.sets.opts_surface_v1 import attach_surface_columns

if TYPE_CHECKING:  # pragma: no cover
    from research.features.feature_registry import FeatureBuildContext

OI_RATIO_MIN = 0.05
OI_RATIO_MAX = 20.0
VOLUME_RATIO_EPS = 1e-6
OPTIONS_SURFACE_V1_COLUMNS: Tuple[str, ...] = (
    "OPT:OI:TOTAL",
    "OPT:OI:CALL_PUT_RATIO",
    "OPT:VOL:TOTAL",
    "OPT:VOL:CALL_PUT_RATIO",
    "OPT:IVX:30",
    "OPT:IVX:90",
    "OPT:IVX:180",
    "OPT:IVR:30",
    "OPT:IVR:90",
    "OPT:COVERAGE:ROW_VALID",
    "OPT:COVERAGE:HAS_OI",
    "OPT:COVERAGE:HAS_OPT_VOLUME",
    "OPT:COVERAGE:HAS_IVX",
    "OPT:COVERAGE:HAS_IVR",
)
DENSE_NUMERIC_COLUMNS: Tuple[str, ...] = OPTIONS_SURFACE_V1_COLUMNS[:9]
_COVERAGE_COLUMNS: Tuple[str, ...] = OPTIONS_SURFACE_V1_COLUMNS[9:]
_COVERAGE_BY_COLUMN: Dict[str, str] = {
    "OPT:OI:TOTAL": "OPT:COVERAGE:HAS_OI",
    "OPT:OI:CALL_PUT_RATIO": "OPT:COVERAGE:HAS_OI",
    "OPT:VOL:TOTAL": "OPT:COVERAGE:HAS_OPT_VOLUME",
    "OPT:VOL:CALL_PUT_RATIO": "OPT:COVERAGE:HAS_OPT_VOLUME",
    "OPT:IVX:30": "OPT:COVERAGE:HAS_IVX",
    "OPT:IVX:90": "OPT:COVERAGE:HAS_IVX",
    "OPT:IVX:180": "OPT:COVERAGE:HAS_IVX",
    "OPT:IVR:30": "OPT:COVERAGE:HAS_IVR",
    "OPT:IVR:90": "OPT:COVERAGE:HAS_IVR",
}


def build_options_surface_v1_features(
    equity_frame: "pd.DataFrame",
    _options: object | None,
    context: FeatureBuildContext,
) -> "pd.DataFrame":
    """Build OPTIONS_SURFACE_V1 with deterministic fill and clipping rules."""

    _ensure_pandas_available()
    if context is None:
        raise ValueError("Feature build context is required for options_surface_v1")
    surface_slice, file_hashes = _load_surface_slice(context)
    context.inputs_used.update(file_hashes)
    merged = attach_surface_columns(equity_frame, surface_slice)
    working = merged.copy()
    working.sort_values("timestamp", inplace=True, kind="mergesort")
    working.reset_index(drop=True, inplace=True)
    _apply_fill_policies(working)
    ordered = ["timestamp", *OPTIONS_SURFACE_V1_COLUMNS]
    return working.loc[:, ordered]


def _load_surface_slice(
    context: FeatureBuildContext,
) -> Tuple[OptionsSurfaceSlice | None, Dict[str, str]]:
    slices, hashes = load_options_surface(
        [context.symbol],
        context.start_date,
        context.end_date,
        data_root=context.data_root,
    )
    return slices.get(context.symbol), hashes


def _apply_fill_policies(frame: "pd.DataFrame") -> None:
    _fill_level_with_window(frame, "OPT:OI:TOTAL")
    _fill_oi_ratio(frame)
    _fill_volume_metrics(frame)
    for column in ("OPT:IVX:30", "OPT:IVX:90", "OPT:IVX:180", "OPT:IVR:30", "OPT:IVR:90"):
        _fill_level_with_window(frame, column)
    _update_coverage_flags(frame)


def _fill_level_with_window(frame: "pd.DataFrame", column: str) -> None:
    if column not in frame:
        return
    series = frame[column].astype(float)
    frame[column] = series.ffill().bfill(limit=2)


def _fill_oi_ratio(frame: "pd.DataFrame") -> None:
    column = "OPT:OI:CALL_PUT_RATIO"
    if column not in frame:
        return
    if {"OPT:OI:CALL", "OPT:OI:PUT"}.issubset(frame.columns):
        call = frame["OPT:OI:CALL"].astype(float)
        put = frame["OPT:OI:PUT"].astype(float)
        denom = put.where(put.abs() > VOLUME_RATIO_EPS)
        ratio = call / denom
        ratio[(call.isna()) | (put.isna())] = np.nan
        mask = frame[column].isna() & ratio.notna()
        frame.loc[mask, column] = ratio[mask]
    filled = frame[column].astype(float).ffill().bfill(limit=2)
    frame[column] = filled.clip(lower=OI_RATIO_MIN, upper=OI_RATIO_MAX)


def _fill_volume_metrics(frame: "pd.DataFrame") -> None:
    total_column = "OPT:VOL:TOTAL"
    if total_column in frame:
        frame[total_column] = frame[total_column].astype(float).fillna(0.0)
    ratio_column = "OPT:VOL:CALL_PUT_RATIO"
    if ratio_column not in frame:
        return
    if {"OPT:VOL:CALL", "OPT:VOL:PUT"}.issubset(frame.columns):
        call = frame["OPT:VOL:CALL"].astype(float)
        put = frame["OPT:VOL:PUT"].astype(float)
        denom = put.clip(lower=VOLUME_RATIO_EPS)
        ratio = call / denom
        ratio[(call.isna()) | (put.isna())] = np.nan
        mask = frame[ratio_column].isna() & ratio.notna()
        frame.loc[mask, ratio_column] = ratio[mask]
    missing_mask = frame[ratio_column].isna()
    if missing_mask.any() and "OPT:COVERAGE:HAS_OPT_VOLUME" in frame:
        has_volume = frame["OPT:COVERAGE:HAS_OPT_VOLUME"].astype(bool)
        has_volume &= ~missing_mask
        frame["OPT:COVERAGE:HAS_OPT_VOLUME"] = has_volume


def _update_coverage_flags(frame: "pd.DataFrame") -> None:
    for column, coverage in _COVERAGE_BY_COLUMN.items():
        if column not in frame or coverage not in frame:
            continue
        missing = frame[column].isna()
        if missing.any():
            availability = frame[coverage].astype(bool)
            frame[coverage] = availability & (~missing)
    if "OPT:COVERAGE:ROW_VALID" in frame:
        valid = frame["OPT:COVERAGE:ROW_VALID"].astype(bool)
        dense_mask = pd.Series(True, index=frame.index)
        for column in DENSE_NUMERIC_COLUMNS:
            if column not in frame:
                continue
            dense_mask &= frame[column].notna()
        frame["OPT:COVERAGE:ROW_VALID"] = valid & dense_mask
    for column in _COVERAGE_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype(bool)


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for options_surface_v1 feature builder") from _PANDAS_ERROR


__all__ = [
    "OPTIONS_SURFACE_V1_COLUMNS",
    "build_options_surface_v1_features",
]
