"""Formatting helpers for dashboard tables and metrics."""
from __future__ import annotations

from typing import Any

from dashboard.config import COLOR_BAD, COLOR_GOOD


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_sharpe(value: Any) -> str:
    val = _to_float(value)
    return "n/a" if val is None else f"{val:.2f}"


def format_pct(value: Any) -> str:
    val = _to_float(value)
    return "n/a" if val is None else f"{val * 100:.2f}%"


def format_pct_1(value: Any) -> str:
    val = _to_float(value)
    return "n/a" if val is None else f"{val * 100:.1f}%"


def format_turnover(value: Any) -> str:
    val = _to_float(value)
    return "n/a" if val is None else f"{val:.2f}"


def format_exposure(value: Any) -> str:
    val = _to_float(value)
    return "n/a" if val is None else f"{val:.2f}"


def format_bps(value: Any) -> str:
    val = _to_float(value)
    return "n/a" if val is None else f"{val:.1f} bps"


def format_delta(metric_key: str, value: Any) -> str:
    if metric_key in {"total_return", "max_drawdown", "volatility_ann"}:
        return _format_signed_pct(value, decimals=2)
    if metric_key in {"tx_cost_bps"}:
        return _format_signed_number(value, decimals=1, suffix=" bps")
    if metric_key in {"sharpe"}:
        return _format_signed_number(value, decimals=2)
    return _format_signed_number(value, decimals=2)


def format_delta_pct(value: Any) -> str:
    return _format_signed_pct(value, decimals=1)


def _format_signed_number(value: Any, *, decimals: int, suffix: str = "") -> str:
    val = _to_float(value)
    if val is None:
        return "n/a"
    sign = "+" if val > 0 else "" if val == 0 else ""
    return f"{sign}{val:.{decimals}f}{suffix}"


def _format_signed_pct(value: Any, *, decimals: int) -> str:
    val = _to_float(value)
    if val is None:
        return "n/a"
    sign = "+" if val > 0 else "" if val == 0 else ""
    return f"{sign}{val * 100:.{decimals}f}%"


def column_label(field: str) -> str:
    mapping = {
        "experiment_name": "Name",
        "experiment_id": "ID",
        "feature_set": "Feature Set",
        "policy": "Policy",
        "sharpe": "Sharpe",
        "total_return": "Total Return",
        "max_drawdown": "Max Drawdown",
        "avg_turnover": "Avg Turnover",
        "avg_exposure": "Avg Exposure",
        "volatility_ann": "Ann. Volatility",
        "turnover_1d_mean": "Turnover",
        "tx_cost_bps": "Tx Cost",
        "delta_pct": "Δ%",
        "delta": "Δ",
        "regime": "Regime",
        "equity_ohlcv": "Equity OHLCV",
        "options_surface": "Options Surface",
        "fundamentals": "Fundamentals",
        "insiders": "Insiders",
    }
    return mapping.get(field, field.replace("_", " ").title())


def delta_color(metric_key: str, delta: Any) -> str | None:
    val = _to_float(delta)
    if val is None:
        return None
    invert = metric_key in {"max_drawdown", "tx_cost_bps"}
    good = val < 0 if invert else val > 0
    bad = val > 0 if invert else val < 0
    if good:
        return COLOR_GOOD
    if bad:
        return COLOR_BAD
    return None


__all__ = [
    "format_sharpe",
    "format_pct",
    "format_pct_1",
    "format_turnover",
    "format_exposure",
    "format_bps",
    "format_delta",
    "format_delta_pct",
    "column_label",
    "delta_color",
]
