"""Streamlit chat interface for transcript-grounded meeting questions."""

from __future__ import annotations

import logging

import requests
import streamlit as st

from app.rag.chat_service import MeetingChatService

logger = logging.getLogger(__name__)
CHAT_KEY = "meeting_chat_service"
MEETING_KEY = "meeting_chat_id"


def _render_turn(role: str, content: str, sources: tuple[str, ...] = ()) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            st.caption("Sources: " + ", ".join(sources))


def render_chat_page(default_meeting_id: str | None = None) -> None:
    """Render the meeting selector, conversation, and clear-history control."""
    st.subheader("Ask about a meeting")
    normalized_default = (default_meeting_id or "").strip()
    if normalized_default and st.session_state.get(
        "last_chat_default_meeting_id"
    ) != normalized_default:
        st.session_state["chat_meeting_id_input"] = normalized_default
    st.session_state["last_chat_default_meeting_id"] = normalized_default
    meeting_id = st.text_input(
        "Meeting ID",
        help="Enter the meeting ID used when the transcript was indexed.",
        key="chat_meeting_id_input",
    ).strip()
    if not meeting_id:
        st.info("Enter a meeting ID to start a transcript-grounded chat.")
        return

    active_meeting = st.session_state.get(MEETING_KEY)
    if active_meeting != meeting_id:
        st.session_state[MEETING_KEY] = meeting_id
        st.session_state[CHAT_KEY] = MeetingChatService(meeting_id)

    chat: MeetingChatService = st.session_state[CHAT_KEY]
    if st.button("Clear chat history", disabled=not chat.history):
        chat.clear_history()
        st.rerun()

    for turn in chat.history:
        _render_turn("user", turn.question)
        _render_turn("assistant", turn.answer, turn.sources)

    question = st.chat_input("Ask a question about this meeting")
    if not question:
        return

    _render_turn("user", question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the transcript and preparing an answer..."):
            try:
                turn = chat.ask(question)
            except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
                logger.exception("Could not answer a meeting chat question.")
                st.error(f"Could not answer this question: {exc}")
                return
        st.markdown(turn.answer)
        if turn.sources:
            st.caption("Sources: " + ", ".join(turn.sources))
