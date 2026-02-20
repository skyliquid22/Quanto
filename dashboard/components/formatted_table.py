"""HTML table renderer with consistent styling."""
from __future__ import annotations

from typing import Iterable, Mapping

import streamlit as st

from dashboard.config import COLOR_MUTED, COLOR_ROW_ALT


def render_table(
    *,
    columns: Iterable[Mapping[str, str]],
    rows: Iterable[Mapping[str, str]],
    right_align: Iterable[str] | None = None,
) -> None:
    right_align = set(right_align or [])
    header_cells = []
    for col in columns:
        label = col["label"]
        align = "right" if col["key"] in right_align else "left"
        header_cells.append(
            f"<th style=\"text-align:{align};padding:6px 10px;"
            f"color:{COLOR_MUTED};border-bottom:1px solid #333;\">{label}</th>"
        )
    header_html = "".join(header_cells)

    body_rows = []
    for idx, row in enumerate(rows):
        bg = COLOR_ROW_ALT if idx % 2 == 1 else "transparent"
        cells = []
        for col in columns:
            key = col["key"]
            align = "right" if key in right_align else "left"
            value = row.get(key, "")
            cells.append(
                f"<td style=\"text-align:{align};padding:6px 10px;\">{value}</td>"
            )
        body_rows.append(f"<tr style=\"background:{bg};\">{''.join(cells)}</tr>")

    html = f"""
    <table style=\"width:100%;border-collapse:collapse;font-size:13px;\">
      <thead><tr>{header_html}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


__all__ = ["render_table"]
