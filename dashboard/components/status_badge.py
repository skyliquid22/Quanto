"""Status badge HTML components."""
from __future__ import annotations

from dashboard.config import COLOR_BAD, COLOR_GOOD, COLOR_WARN


def render_badge(label: str, kind: str) -> str:
    kind_lower = kind.lower()
    if kind_lower in {"pass", "qualified", "good"}:
        bg = "#1E3A2F"
        fg = COLOR_GOOD
    elif kind_lower in {"fail", "not_qualified", "bad"}:
        bg = "#5C2626"
        fg = COLOR_BAD
    else:
        bg = "#3D3200"
        fg = COLOR_WARN
    return (
        f"<span style=\"background:{bg};color:{fg};padding:2px 8px;"
        "border-radius:3px;font-size:12px;\">"
        f"{label}</span>"
    )


__all__ = ["render_badge"]
