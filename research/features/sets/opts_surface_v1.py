"""Helper utilities for joining options surface columns into feature frames."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:  # pragma: no cover - pandas optional
    import pandas as pd  # type: ignore
except Exception as exc:  # pragma: no cover
    pd = None
    _PANDAS_ERROR: Exception | None = exc
else:  # pragma: no cover
    _PANDAS_ERROR = None

from research.datasets.options_surface_loader import OptionsSurfaceSlice
from infra.normalization.options_surface import OPT_COVERAGE_COLUMNS, OPT_SURFACE_ALL_OPT_COLUMNS

OPTIONS_SURFACE_FEATURE_COLUMNS: tuple[str, ...] = tuple(sorted(OPT_SURFACE_ALL_OPT_COLUMNS))
_COVERAGE_SET = set(OPT_COVERAGE_COLUMNS)


def attach_surface_columns(
    equity_frame: "pd.DataFrame",
    surface_slice: OptionsSurfaceSlice | None,
) -> "pd.DataFrame":
    """Left-join per-date surface metrics onto a single-symbol equity frame."""

    _ensure_pandas_available()
    frame = equity_frame.copy()
    if "timestamp" not in frame:
        raise ValueError("equity_frame must include timestamp columns to join surface data")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["_surface_date"] = frame["timestamp"].dt.normalize()

    if surface_slice is None or surface_slice.frame.empty:
        for column in OPTIONS_SURFACE_FEATURE_COLUMNS:
            if column in _COVERAGE_SET:
                frame[column] = False
            else:
                frame[column] = np.nan
        frame.drop(columns=["_surface_date"], inplace=True)
        return frame

    surface = surface_slice.frame.copy()
    if "date" not in surface:
        raise ValueError("surface slices must include date columns")
    surface["_surface_date"] = pd.to_datetime(surface["date"], utc=True)
    surface.sort_values("_surface_date", inplace=True, kind="mergesort")
    surface = surface.drop_duplicates(subset="_surface_date", keep="last")
    columns = ["_surface_date"] + [column for column in OPTIONS_SURFACE_FEATURE_COLUMNS if column in surface.columns]
    merged = frame.merge(surface[columns], on="_surface_date", how="left")
    for column in OPTIONS_SURFACE_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = False if column in _COVERAGE_SET else np.nan
    coverage_columns = [column for column in OPTIONS_SURFACE_FEATURE_COLUMNS if column in _COVERAGE_SET]
    if coverage_columns:
        coverage_block = merged[coverage_columns].astype("boolean")
        coverage_block = coverage_block.fillna(False)
        merged.loc[:, coverage_columns] = coverage_block.astype(bool, copy=False)

    value_columns = [column for column in OPTIONS_SURFACE_FEATURE_COLUMNS if column not in _COVERAGE_SET]
    if value_columns:
        merged.loc[:, value_columns] = merged[value_columns].apply(pd.to_numeric, errors="coerce")
    merged.drop(columns=["_surface_date"], inplace=True)
    return merged


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for options surface feature joins") from _PANDAS_ERROR


__all__ = ["OPTIONS_SURFACE_FEATURE_COLUMNS", "attach_surface_columns"]
