"""Monitor placeholder page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(_: Path) -> None:
    st.header("Monitor")
    st.caption("Pipeline monitoring, drift detection, and alerting — coming in a future release.")


__all__ = ["render"]
