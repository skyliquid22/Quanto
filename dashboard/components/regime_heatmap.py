"""Regime heatmap HTML component."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config import HEATMAP_GREEN, HEATMAP_NEUTRAL, HEATMAP_RED
from dashboard.styles import column_label, format_exposure, format_pct, format_sharpe, format_turnover


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _lerp_color(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    return _rgb_to_hex(tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3)))


def _color_for_value(value: float, min_val: float, max_val: float, invert: bool) -> str:
    if max_val == min_val:
        t = 0.5
    else:
        t = (value - min_val) / (max_val - min_val)
    if invert:
        t = 1.0 - t
    red = _hex_to_rgb(HEATMAP_RED)
    neutral = _hex_to_rgb(HEATMAP_NEUTRAL)
    green = _hex_to_rgb(HEATMAP_GREEN)
    if t <= 0.5:
        return _lerp_color(red, neutral, t / 0.5)
    return _lerp_color(neutral, green, (t - 0.5) / 0.5)


def _format_metric(col: str, value: float | None) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    if col in {"sharpe"}:
        return format_sharpe(value)
    if col in {"max_drawdown", "total_return", "volatility_ann"}:
        return format_pct(value)
    if col in {"avg_turnover"}:
        return format_turnover(value)
    if col in {"avg_exposure"}:
        return format_exposure(value)
    return f"{value:.4f}"


def render_regime_heatmap(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    invert_cols: Iterable[str] = ("max_drawdown", "volatility_ann"),
) -> None:
    if frame.empty:
        st.info("Regime slices not available for this experiment.")
        return
    invert_set = set(invert_cols)
    columns = list(columns)

    stats = {}
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        stats[col] = (series.min(skipna=True), series.max(skipna=True))

    header_cells = [
        "<th style=\"text-align:left;padding:6px 10px;color:#888;border-bottom:1px solid #333;\">Regime</th>"
    ]
    for col in columns:
        header_cells.append(
            f"<th style=\"text-align:right;padding:6px 10px;color:#888;border-bottom:1px solid #333;\">{column_label(col)}</th>"
        )

    body_rows = []
    for idx, (regime, row) in enumerate(frame.iterrows()):
        bg = "rgba(255,255,255,0.03)" if idx % 2 == 1 else "transparent"
        cells = [
            f"<td style=\"text-align:left;padding:6px 10px;\">{regime}</td>"
        ]
        for col in columns:
            value = row.get(col)
            formatted = _format_metric(col, value)
            min_val, max_val = stats[col]
            color = HEATMAP_NEUTRAL
            if value is not None and not np.isnan(value) and min_val is not None and max_val is not None:
                color = _color_for_value(float(value), float(min_val), float(max_val), col in invert_set)
            cells.append(
                f"<td style=\"text-align:right;padding:6px 10px;background:{color};color:#E0E0E0;\">{formatted}</td>"
            )
        body_rows.append(f"<tr style=\"background:{bg};\">{''.join(cells)}</tr>")

    html = f"""
    <table style=\"width:100%;border-collapse:collapse;font-size:13px;\">
      <thead><tr>{''.join(header_cells)}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


__all__ = ["render_regime_heatmap"]
