"""Main entry point for the institutional-grade Quanto dashboard."""
from __future__ import annotations

import streamlit as st

from dashboard.config import resolve_data_root
from dashboard.pages import execute, monitor, research, status


def _inject_css() -> None:
    st.markdown(
        """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
.stDataFrame tbody tr:nth-child(even) {background: rgba(255,255,255,0.03);}
</style>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Quanto Dashboard", layout="wide")
    _inject_css()

    data_root = resolve_data_root()

    pages = [
        st.Page(lambda: status.render(data_root), title="Status"),
        st.Page(lambda: research.render(data_root), title="Research"),
        st.Page(lambda: monitor.render(data_root), title="Monitor"),
        st.Page(lambda: execute.render(data_root), title="Execute"),
    ]

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
