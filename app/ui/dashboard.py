"""Streamlit workflow for transcribing, indexing, and analyzing meetings."""

from __future__ import annotations

import logging

import requests
import streamlit as st

from app.agents.action_item_extractor import extract_action_items
from app.agents.decision_extractor import extract_decisions
from app.agents.summarizer import summarize_meeting
from app.rag.document_loader import load_transcript
from app.rag.text_splitter import split_transcript
from app.rag.vector_store import index_documents
from app.transcription.audio_processor import SUPPORTED_MEDIA_EXTENSIONS
from app.transcription.models import TranscriptionResult
from app.transcription.whisper_service import transcribe_audio
from app.transcription.youtube_service import fetch_youtube_captions
from app.utils.export_utils import (
    ExportError,
    export_action_items_pdf,
    export_action_items_txt,
    export_summary_pdf,
    export_summary_txt,
    export_transcript_txt,
)
from app.utils.file_handler import (
    StoredMedia,
    create_meeting_id,
    save_transcript,
    save_uploaded_media,
)

logger = logging.getLogger(__name__)
EXPECTED_PROCESSING_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    requests.RequestException,
)


def _activate_transcript(
    meeting_id: str,
    source_name: str,
    transcription: TranscriptionResult,
) -> None:
    save_transcript(meeting_id, transcription.text)
    st.session_state["active_meeting_id"] = meeting_id
    st.session_state["current_source_name"] = source_name
    st.session_state["current_transcription"] = transcription
    st.session_state["meeting_indexed"] = False
    for key in ("meeting_summary", "meeting_decisions", "meeting_action_items"):
        st.session_state.pop(key, None)


def _render_upload_tab() -> None:
    extensions = sorted(
        extension.removeprefix(".") for extension in SUPPORTED_MEDIA_EXTENSIONS
    )
    uploaded_file = st.file_uploader(
        "Upload an audio or video recording",
        type=extensions,
        help="The maximum upload size is configured by MAX_UPLOAD_SIZE_MB.",
    )
    if uploaded_file is None:
        return

    st.caption(f"Selected: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
    if st.button("Transcribe recording", type="primary"):
        try:
            with st.status("Transcribing recording...", expanded=True) as status:
                status.write("Saving the uploaded recording.")
                stored: StoredMedia = save_uploaded_media(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                status.write("Running local Whisper transcription.")
                transcription = transcribe_audio(stored.path)
                _activate_transcript(
                    stored.meeting_id,
                    stored.original_name,
                    transcription,
                )
                status.update(label="Transcription complete.", state="complete")
        except EXPECTED_PROCESSING_ERRORS as exc:
            logger.exception("Could not transcribe uploaded media.")
            st.error(f"Could not transcribe this recording: {exc}")


def _render_youtube_tab() -> None:
    url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Uses captions already provided by YouTube; it does not download the video.",
    )
    if st.button("Load available captions", type="primary", disabled=not url.strip()):
        try:
            with st.status("Retrieving available captions...", expanded=True) as status:
                transcription = fetch_youtube_captions(url)
                meeting_id = create_meeting_id()
                _activate_transcript(
                    meeting_id,
                    url.strip(),
                    transcription,
                )
                status.update(label="Captions loaded.", state="complete")
        except EXPECTED_PROCESSING_ERRORS as exc:
            logger.exception("Could not retrieve YouTube captions.")
            st.error(f"Could not load captions: {exc}")


def _render_transcript() -> None:
    transcription = st.session_state.get("current_transcription")
    if not isinstance(transcription, TranscriptionResult):
        return

    meeting_id = st.session_state["active_meeting_id"]
    st.subheader("Transcript")
    st.caption(
        f"Meeting ID: `{meeting_id}` | Source: "
        f"{st.session_state.get('current_source_name', 'transcript')} | "
        f"Language: {transcription.language or 'unknown'}"
    )
    st.text_area(
        "Transcript text",
        value=transcription.text,
        height=280,
        disabled=True,
        label_visibility="collapsed",
    )

    if not st.session_state.get("meeting_indexed", False):
        if st.button("Index transcript for meeting chat"):
            try:
                with st.status("Indexing transcript...", expanded=True) as status:
                    documents = load_transcript(
                        transcription,
                        meeting_id=meeting_id,
                    )
                    chunks = split_transcript(documents)
                    status.write(f"Adding {len(chunks)} transcript chunks to Chroma.")
                    index_documents(chunks, meeting_id=meeting_id)
                    st.session_state["meeting_indexed"] = True
                    status.update(label="Transcript indexed.", state="complete")
            except EXPECTED_PROCESSING_ERRORS as exc:
                logger.exception("Could not index meeting transcript.")
                st.error(f"Could not index this transcript: {exc}")
    else:
        st.success("Transcript is indexed and ready for meeting chat.")

    _render_insights(transcription.text)


def _render_insights(transcript: str) -> None:
    if st.button("Generate meeting insights"):
        try:
            with st.status("Generating meeting insights...", expanded=True) as status:
                status.write("Creating the structured meeting summary.")
                st.session_state["meeting_summary"] = summarize_meeting(transcript)
                status.write("Extracting explicit decisions.")
                st.session_state["meeting_decisions"] = extract_decisions(transcript)
                status.write("Extracting action items.")
                st.session_state["meeting_action_items"] = extract_action_items(transcript)
                status.update(label="Meeting insights generated.", state="complete")
        except EXPECTED_PROCESSING_ERRORS as exc:
            logger.exception("Could not generate meeting insights.")
            st.error(f"Could not generate all meeting insights: {exc}")

    summary = st.session_state.get("meeting_summary")
    if summary is not None:
        st.subheader("Meeting summary")
        st.markdown(summary.executive_summary)
        with st.expander("Detailed summary and discussion topics"):
            st.markdown(summary.detailed_summary)
            st.markdown("**Main topics**")
            for topic in summary.main_topics:
                st.markdown(f"- {topic}")
            st.markdown("**Unresolved questions**")
            for question in summary.unresolved_questions:
                st.markdown(f"- {question}")
            st.markdown("**Follow-up requirements**")
            for requirement in summary.follow_up_requirements:
                st.markdown(f"- {requirement}")

    decisions = st.session_state.get("meeting_decisions")
    if decisions is not None:
        st.subheader("Decisions")
        if decisions.decisions:
            for decision in decisions.decisions:
                st.markdown(f"- {decision}")
        else:
            st.write("No explicit decisions were found.")

    action_items = st.session_state.get("meeting_action_items")
    if action_items is not None:
        st.subheader("Action items")
        if action_items.action_items:
            st.dataframe(
                [item.model_dump() for item in action_items.action_items],
                width="stretch",
                hide_index=True,
            )
        else:
            st.write("No explicit action items were found.")


def _render_exports(transcription: TranscriptionResult) -> None:
    meeting_id = st.session_state["active_meeting_id"]
    st.subheader("Exports")
    st.download_button(
        "Download transcript TXT",
        data=export_transcript_txt(transcription),
        file_name=f"meeting_{meeting_id}_transcript.txt",
        mime="text/plain",
        key=f"transcript_txt_{meeting_id}",
    )

    summary = st.session_state.get("meeting_summary")
    if summary is not None:
        try:
            summary_pdf = export_summary_pdf(
                summary,
                decisions=st.session_state.get("meeting_decisions"),
                action_items=st.session_state.get("meeting_action_items"),
            )
        except ExportError as exc:
            st.error(f"Could not generate the summary PDF: {exc}")
        else:
            first_column, second_column = st.columns(2)
            with first_column:
                st.download_button(
                    "Download summary PDF",
                    data=summary_pdf,
                    file_name=f"meeting_{meeting_id}_summary.pdf",
                    mime="application/pdf",
                    key=f"summary_pdf_{meeting_id}",
                )
            with second_column:
                st.download_button(
                    "Download summary TXT",
                    data=export_summary_txt(
                        summary,
                        decisions=st.session_state.get("meeting_decisions"),
                        action_items=st.session_state.get("meeting_action_items"),
                    ),
                    file_name=f"meeting_{meeting_id}_summary.txt",
                    mime="text/plain",
                    key=f"summary_txt_{meeting_id}",
                )

    action_items = st.session_state.get("meeting_action_items")
    if action_items is not None:
        try:
            action_items_pdf = export_action_items_pdf(action_items)
        except ExportError as exc:
            st.error(f"Could not generate the action-items PDF: {exc}")
        else:
            first_column, second_column = st.columns(2)
            with first_column:
                st.download_button(
                    "Download action items PDF",
                    data=action_items_pdf,
                    file_name=f"meeting_{meeting_id}_action_items.pdf",
                    mime="application/pdf",
                    key=f"action_items_pdf_{meeting_id}",
                )
            with second_column:
                st.download_button(
                    "Download action items TXT",
                    data=export_action_items_txt(action_items),
                    file_name=f"meeting_{meeting_id}_action_items.txt",
                    mime="text/plain",
                    key=f"action_items_txt_{meeting_id}",
                )


def render_dashboard() -> None:
    """Render the meeting intake, transcript, index, and insights workflow."""
    st.header("Process a meeting")
    upload_tab, youtube_tab = st.tabs(["Audio/video upload", "YouTube captions"])
    with upload_tab:
        _render_upload_tab()
    with youtube_tab:
        _render_youtube_tab()
    _render_transcript()
    transcription = st.session_state.get("current_transcription")
    if isinstance(transcription, TranscriptionResult):
        _render_exports(transcription)
