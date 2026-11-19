from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.agents.action_item_extractor import ActionItem, ActionItemList
from app.agents.decision_extractor import DecisionList
from app.agents.summarizer import MeetingSummary
from app.transcription.models import TranscriptionResult


def test_dashboard_and_meeting_chat_render_without_processing_external_services() -> None:
    app_path = Path(__file__).parent.parent / "app" / "main.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert [item.value for item in app.header] == ["Process a meeting"]
    assert [item.label for item in app.text_input] == ["YouTube video URL"]
    assert len(app.get("file_uploader")) == 1

    app.sidebar.radio[0].set_value("Meeting chat").run(timeout=30)

    assert not app.exception
    assert [item.label for item in app.text_input] == ["Meeting ID"]


def test_dashboard_exposes_report_downloads_when_meeting_data_exists() -> None:
    app_path = Path(__file__).parent.parent / "app" / "main.py"
    app = AppTest.from_file(str(app_path))
    app.session_state["active_meeting_id"] = "a" * 32
    app.session_state["current_source_name"] = "meeting.wav"
    app.session_state["current_transcription"] = TranscriptionResult(
        text="Team agreed to ship Friday.",
        language="en",
        segments=(),
    )
    app.session_state["meeting_summary"] = MeetingSummary(
        executive_summary="Ship Friday.",
        detailed_summary="The team agreed to ship Friday.",
    )
    app.session_state["meeting_decisions"] = DecisionList(
        decisions=["Ship Friday."]
    )
    app.session_state["meeting_action_items"] = ActionItemList(
        action_items=[ActionItem(task="Send release notes")]
    )

    app.run(timeout=30)

    assert not app.exception
    download_labels = [item.label for item in app.get("download_button")]
    assert "Download transcript TXT" in download_labels
    assert "Download summary PDF" in download_labels
    assert "Download summary TXT" in download_labels
    assert "Download action items PDF" in download_labels
    assert "Download action items TXT" in download_labels
