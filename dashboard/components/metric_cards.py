"""Metric card helper for KPI strips."""
from __future__ import annotations

from typing import Any, Sequence

import streamlit as st

from dashboard.styles import (
    format_bps,
    format_exposure,
    format_pct,
    format_sharpe,
    format_turnover,
)


def _format_metric(metric_key: str, value: Any) -> str:
    if metric_key == "sharpe":
        return format_sharpe(value)
    if metric_key in {"total_return", "max_drawdown"}:
        return format_pct(value)
    if metric_key == "turnover_1d_mean":
        return format_turnover(value)
    if metric_key == "avg_exposure":
        return format_exposure(value)
    if metric_key == "tx_cost_bps":
        return format_bps(value)
    return "n/a"


def render_kpi_strip(metrics: Sequence[dict[str, Any]], baseline: dict[str, Any] | None = None) -> None:
    columns = st.columns(len(metrics))
    for col, metric in zip(columns, metrics):
        key = metric["key"]
        value = _format_metric(key, metric.get("value"))
        delta_value = None
        delta_color = "normal"
        if baseline and key in baseline:
            delta = metric.get("value")
            base = baseline.get(key)
            if delta is not None and base is not None:
                delta_value = _format_metric(key, delta - base)
        if key == "max_drawdown":
            delta_color = "inverse"
        col.metric(metric["label"], value=value, delta=delta_value, delta_color=delta_color)


__all__ = ["render_kpi_strip"]
