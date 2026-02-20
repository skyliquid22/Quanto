"""Execute placeholder page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(_: Path) -> None:
    st.header("Execute")
    st.caption("Shadow replay, paper trading, and live execution controls — coming in a future release.")


__all__ = ["render"]
