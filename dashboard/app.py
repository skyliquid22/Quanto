"""Main entry point for the institutional-grade Quanto dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure repo root is on sys.path even when Streamlit is launched outside the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.config import resolve_data_root
from dashboard.pages import execute, monitor, research, status


def _status_page(data_root: Path) -> None:
    status.render(data_root)


def _research_page(data_root: Path) -> None:
    research.render(data_root)


def _monitor_page(data_root: Path) -> None:
    monitor.render(data_root)


def _execute_page(data_root: Path) -> None:
    execute.render(data_root)


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
        st.Page(lambda: _status_page(data_root), title="Status", url_path="status"),
        st.Page(lambda: _research_page(data_root), title="Research", url_path="research"),
        st.Page(lambda: _monitor_page(data_root), title="Monitor", url_path="monitor"),
        st.Page(lambda: _execute_page(data_root), title="Execute", url_path="execute"),
    ]

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
