"""Streamlit entry point for the AI Meeting Assistant."""

from __future__ import annotations

import streamlit as st

from app.config import ensure_runtime_directories
from app.ui.chat_page import render_chat_page
from app.ui.dashboard import render_dashboard


def main() -> None:
    """Render the initial application shell."""
    ensure_runtime_directories()
    st.set_page_config(
        page_title="AI Meeting Assistant",
        page_icon=":speech_balloon:",
        layout="wide",
    )
    st.title("AI Meeting Assistant with RAG")
    page = st.sidebar.radio("Workspace", ["Dashboard", "Meeting chat"])
    if page == "Dashboard":
        render_dashboard()
    else:
        render_chat_page(st.session_state.get("active_meeting_id"))


if __name__ == "__main__":
    main()