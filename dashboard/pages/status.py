"""Status page."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from dashboard.components.formatted_table import render_table
from dashboard.components.status_badge import render_badge
from dashboard.config import COLOR_BAD, COLOR_GOOD, COLOR_WARN
from dashboard.data import (
    load_domain_coverage,
    load_experiment_summaries,
    load_latest_data_health,
    load_recent_promotions,
)
from dashboard.styles import (
    column_label,
    format_pct_1,
    format_sharpe,
)


def _human_bytes(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    size = float(size_bytes)
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _coverage_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pct >= 95:
        color = COLOR_GOOD
    elif pct >= 80:
        color = COLOR_WARN
    else:
        color = COLOR_BAD
    return f"<span style=\"color:{color};\">{pct:.1f}%</span>"


def _health_badge(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return render_badge("UNKNOWN", "warn")
    canonical = payload.get("canonical_summary") or {}
    missing_ratio = canonical.get("overall", {}).get("missing_ratio")
    if missing_ratio is None:
        return render_badge("WARN", "warn")
    missing_ratio = float(missing_ratio)
    if missing_ratio > 0.05:
        return render_badge("FAIL", "fail")
    if missing_ratio > 0.01:
        return render_badge("WARN", "warn")
    return render_badge("PASS", "pass")


def render(data_root: Path) -> None:
    st.header("Status")

    with st.container():
        st.subheader("Data Infrastructure")
        size_cols = st.columns(3)
        size_cols[0].metric("Raw", _human_bytes(_dir_size(data_root / "raw")))
        size_cols[1].metric("Canonical", _human_bytes(_dir_size(data_root / "canonical")))
        size_cols[2].metric("Derived", _human_bytes(_dir_size(data_root / "derived")))

        coverage = load_domain_coverage(data_root)
        rows = []
        for domain in ("equity_ohlcv", "options_surface", "fundamentals", "insiders"):
            entry = coverage.get(domain, {})
            rows.append(
                {
                    "domain": column_label(domain),
                    "min": entry.get("min_date") or "n/a",
                    "max": entry.get("max_date") or "n/a",
                    "symbols": str(entry.get("symbols", 0)),
                    "coverage": _coverage_cell(entry.get("coverage")),
                }
            )
        render_table(
            columns=[
                {"key": "domain", "label": "Domain"},
                {"key": "min", "label": "Min Date"},
                {"key": "max", "label": "Max Date"},
                {"key": "symbols", "label": "Symbols"},
                {"key": "coverage", "label": "Coverage %"},
            ],
            rows=rows,
            right_align=["symbols", "coverage"],
        )

        health_payload = load_latest_data_health(data_root)
        timestamp = None
        if health_payload and health_payload.get("__mtime"):
            timestamp = datetime.fromtimestamp(float(health_payload["__mtime"])).isoformat()
        health_cols = st.columns(2)
        health_cols[0].metric("Last Health Check", timestamp or "n/a")
        health_cols[1].markdown(_health_badge(health_payload), unsafe_allow_html=True)

    st.divider()

    with st.container():
        st.subheader("Activity Feed")
        summaries = load_experiment_summaries(data_root)
        recent = summaries[-10:]
        rows = []
        for entry in recent[::-1]:
            status = "pending"
            if entry.qualification_passed is True:
                status = "qualified"
            elif entry.qualification_passed is False:
                status = "not_qualified"
            rows.append(
                {
                    "name": entry.experiment_name or "unknown",
                    "policy": entry.policy or "n/a",
                    "range": f"{entry.start_date or 'n/a'} → {entry.end_date or 'n/a'}",
                    "sharpe": format_sharpe(entry.sharpe),
                    "status": render_badge(status.upper(), status),
                }
            )
        render_table(
            columns=[
                {"key": "name", "label": "Name"},
                {"key": "policy", "label": "Policy"},
                {"key": "range", "label": "Date Range"},
                {"key": "sharpe", "label": "Sharpe"},
                {"key": "status", "label": "Status"},
            ],
            rows=rows,
            right_align=["sharpe"],
        )

        promotions = load_recent_promotions(data_root)
        if promotions:
            rows = []
            for promo in promotions[:5]:
                rows.append(
                    {
                        "name": promo.get("experiment_name") or promo.get("experiment_id", "unknown"),
                        "tier": promo.get("tier", "n/a"),
                        "date": promo.get("created_at", "n/a"),
                        "reason": promo.get("status", "n/a"),
                    }
                )
            render_table(
                columns=[
                    {"key": "name", "label": "Name"},
                    {"key": "tier", "label": "Tier"},
                    {"key": "date", "label": "Date"},
                    {"key": "reason", "label": "Reason"},
                ],
                rows=rows,
                right_align=[],
            )
        else:
            st.info("No promotions found.")


__all__ = ["render"]
