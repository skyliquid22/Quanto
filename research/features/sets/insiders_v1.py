"""Insider trades feature builder (insiders_v1)."""

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

from research.datasets.canonical_insiders_loader import load_canonical_insiders

if TYPE_CHECKING:  # pragma: no cover
    from research.features.feature_registry import FeatureBuildContext


INSIDERS_V1_COLUMNS: Tuple[str, ...] = (
    "insider_filtered_bsr_30d",
    "insider_cluster_score_10d",
    "insider_intensity_pctl_30d",
    "insider_ownership_delta_30d",
)

_WINDOW_BSR = 30
_WINDOW_CLUSTER = 10
_WINDOW_OWNERSHIP = 30
_WINDOW_INTENSITY = 30
_HISTORY_SESSIONS = 756
_LOOKBACK_DAYS = 820


def build_insiders_v1_features(
    equity_frame: "pd.DataFrame",
    context: FeatureBuildContext,
) -> "pd.DataFrame":
    _ensure_pandas_available()
    if context is None:
        raise ValueError("Feature build context is required for insiders_v1")
    slices, hashes = load_canonical_insiders(
        [context.symbol],
        context.start_date,
        context.end_date,
        data_root=context.data_root,
        lookback_days=_LOOKBACK_DAYS,
    )
    context.inputs_used.update(hashes)
    slice_data = slices.get(context.symbol)
    insiders = slice_data.frame.copy() if slice_data else pd.DataFrame()
    return _attach_insiders(equity_frame, insiders)


def _attach_insiders(equity_frame: "pd.DataFrame", insiders: "pd.DataFrame") -> "pd.DataFrame":
    equity = equity_frame.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    equity.sort_values("timestamp", inplace=True, kind="mergesort")
    equity.reset_index(drop=True, inplace=True)
    equity["session_date"] = equity["timestamp"].dt.normalize()

    if "volume" not in equity.columns:
        raise ValueError("insiders_v1 requires equity volume to compute dollar-volume thresholds")

    equity["close"] = equity["close"].astype(float)
    equity["volume"] = equity["volume"].astype(float)
    equity["dollar_volume"] = equity["close"] * equity["volume"]
    equity["avg_daily_dollar_volume_20d"] = equity["dollar_volume"].rolling(window=20, min_periods=20).mean()

    if insiders.empty:
        return _assign_empty_insiders(equity)

    insiders = _prepare_insiders(insiders)
    if insiders.empty:
        return _assign_empty_insiders(equity)

    session_dates = equity["session_date"]
    session_map = {date: idx for idx, date in enumerate(session_dates)}
    avg_map = dict(zip(session_dates, equity["avg_daily_dollar_volume_20d"]))

    insiders["session_idx"] = insiders["filing_date"].map(session_map)
    insiders["avg_daily_dollar_volume_20d"] = insiders["filing_date"].map(avg_map)
    insiders = insiders.dropna(subset=["session_idx", "avg_daily_dollar_volume_20d"])
    if insiders.empty:
        return _assign_empty_insiders(equity)

    insiders["session_idx"] = insiders["session_idx"].astype(int)
    insiders = insiders[insiders["avg_daily_dollar_volume_20d"] > 0]

    title_upper = insiders["title"].fillna("").str.upper()
    role_match = title_upper.str.contains("CEO|CFO|COO|CTO|PRESIDENT|CHAIRMAN", regex=True)
    board_flag = insiders["is_board_director"].fillna(False).astype(bool)
    role_allowed = role_match | board_flag
    security_match = insiders["security_title"] == "Common Stock"

    insiders = insiders[role_allowed & security_match]
    insiders = insiders.dropna(subset=["transaction_value", "transaction_shares"])
    if insiders.empty:
        return _assign_empty_insiders(equity)

    min_value = insiders["avg_daily_dollar_volume_20d"] * 0.001
    insiders = insiders[insiders["transaction_value"] >= min_value]
    if insiders.empty:
        return _assign_empty_insiders(equity)

    title_upper = insiders["title"].fillna("").str.upper()
    board_flag = insiders["is_board_director"].fillna(False).astype(bool)
    high_role = title_upper.str.contains("CEO|CFO|CHAIRMAN", regex=True)
    role_weight = np.where(high_role, 3.0, np.where(board_flag, 2.0, 1.0))
    insiders = insiders.assign(role_weight=role_weight)

    n_sessions = len(equity)
    buy_series = pd.Series(0.0, index=range(n_sessions))
    sell_series = pd.Series(0.0, index=range(n_sessions))
    score_series = pd.Series(0.0, index=range(n_sessions))

    buy_mask = insiders["transaction_shares"] > 0
    sell_mask = insiders["transaction_shares"] < 0

    if buy_mask.any():
        grouped = insiders.loc[buy_mask].groupby("session_idx")["transaction_value"].sum()
        buy_series.loc[grouped.index] = grouped.values
        scores = insiders.loc[buy_mask].copy()
        scores["score"] = scores["role_weight"] * (scores["transaction_value"] / scores["avg_daily_dollar_volume_20d"])
        grouped_scores = scores.groupby("session_idx")["score"].sum()
        score_series.loc[grouped_scores.index] = grouped_scores.values

    if sell_mask.any():
        grouped = insiders.loc[sell_mask].groupby("session_idx")["transaction_value"].sum()
        sell_series.loc[grouped.index] = grouped.values

    buy_30 = buy_series.rolling(window=_WINDOW_BSR, min_periods=1).sum()
    sell_30 = sell_series.rolling(window=_WINDOW_BSR, min_periods=1).sum()
    bsr = (buy_30 - sell_30) / (buy_30 + sell_30 + 1.0)

    cluster_score = score_series.rolling(window=_WINDOW_CLUSTER, min_periods=1).sum()

    ownership_delta = _compute_ownership_delta(insiders, n_sessions)
    intensity = _compute_intensity_percentile(buy_series, n_sessions)

    equity["insider_filtered_bsr_30d"] = bsr.values
    equity["insider_cluster_score_10d"] = cluster_score.values
    equity["insider_intensity_pctl_30d"] = intensity
    equity["insider_ownership_delta_30d"] = ownership_delta

    equity.drop(columns=["session_date", "dollar_volume", "avg_daily_dollar_volume_20d"], inplace=True, errors="ignore")
    return equity


def _compute_ownership_delta(insiders: "pd.DataFrame", n_sessions: int) -> np.ndarray:
    rows_by_session = [[] for _ in range(n_sessions)]
    for _, row in insiders.iterrows():
        idx = int(row["session_idx"])
        rows_by_session[idx].append(row)

    output = np.zeros(n_sessions, dtype="float64")
    for t in range(n_sessions):
        start = max(0, t - _WINDOW_OWNERSHIP + 1)
        window_rows = []
        for idx in range(start, t + 1):
            window_rows.extend(rows_by_session[idx])
        if not window_rows:
            output[t] = 0.0
            continue
        per_insider: Dict[str, Dict[str, Tuple[int, float]]] = {}
        for row in window_rows:
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            before = row.get("shares_owned_before_transaction")
            after = row.get("shares_owned_after_transaction")
            if before is None or after is None or np.isnan(before) or np.isnan(after):
                continue
            before_val = float(before)
            after_val = float(after)
            entry = per_insider.get(name)
            if entry is None:
                per_insider[name] = {
                    "before": (int(row["session_idx"]), before_val),
                    "after": (int(row["session_idx"]), after_val),
                }
                continue
            if int(row["session_idx"]) < entry["before"][0]:
                entry["before"] = (int(row["session_idx"]), before_val)
            if int(row["session_idx"]) > entry["after"][0]:
                entry["after"] = (int(row["session_idx"]), after_val)

        deltas = []
        for entry in per_insider.values():
            before_val = entry["before"][1]
            after_val = entry["after"][1]
            if before_val <= 0:
                continue
            deltas.append((after_val - before_val) / before_val)
        output[t] = float(np.median(deltas)) if deltas else 0.0
    return output


def _compute_intensity_percentile(buy_series: "pd.Series", n_sessions: int) -> np.ndarray:
    output = np.full(n_sessions, np.nan, dtype="float64")
    buy_30 = buy_series.rolling(window=_WINDOW_INTENSITY, min_periods=1).sum()
    for t in range(n_sessions):
        hist_end = t - _WINDOW_INTENSITY
        if hist_end < 0:
            continue
        hist_start = hist_end - _HISTORY_SESSIONS + 1
        if hist_start < 0:
            continue
        window = buy_series.iloc[hist_start : hist_end + 1]
        blocks = len(window) // _WINDOW_INTENSITY
        if blocks <= 0:
            continue
        hist_values = []
        for idx in range(blocks):
            block = window.iloc[idx * _WINDOW_INTENSITY : (idx + 1) * _WINDOW_INTENSITY]
            hist_values.append(float(block.sum()))
        current_value = float(buy_30.iloc[t])
        if current_value == 0.0 and all(value == 0.0 for value in hist_values):
            output[t] = 0.5
            continue
        hist_array = np.asarray(hist_values, dtype="float64")
        output[t] = float(np.mean(hist_array <= current_value))
    return output


def _assign_empty_insiders(frame: "pd.DataFrame") -> "pd.DataFrame":
    for column in INSIDERS_V1_COLUMNS:
        if column == "insider_intensity_pctl_30d":
            frame[column] = 0.5
        else:
            frame[column] = 0.0
    return frame


def _prepare_insiders(frame: "pd.DataFrame") -> "pd.DataFrame":
    data = frame.copy()
    data["filing_date"] = pd.to_datetime(data.get("filing_date"), utc=True, errors="coerce").dt.normalize()
    data.dropna(subset=["filing_date"], inplace=True)
    data["transaction_date"] = pd.to_datetime(data.get("transaction_date"), utc=True, errors="coerce").dt.normalize()
    data["transaction_value"] = pd.to_numeric(data.get("transaction_value"), errors="coerce")
    data["transaction_shares"] = pd.to_numeric(data.get("transaction_shares"), errors="coerce")
    data["shares_owned_before_transaction"] = pd.to_numeric(
        data.get("shares_owned_before_transaction"), errors="coerce"
    )
    data["shares_owned_after_transaction"] = pd.to_numeric(
        data.get("shares_owned_after_transaction"), errors="coerce"
    )
    data["security_title"] = data.get("security_title").astype(str) if "security_title" in data else ""
    data["title"] = data.get("title").astype(str) if "title" in data else ""
    data["name"] = data.get("name").astype(str) if "name" in data else ""
    if "is_board_director" in data:
        data["is_board_director"] = data["is_board_director"].fillna(False).astype(bool)
    else:
        data["is_board_director"] = False
    return data


def _ensure_pandas_available() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for insiders_v1 feature computation") from _PANDAS_ERROR


__all__ = ["INSIDERS_V1_COLUMNS", "build_insiders_v1_features"]
