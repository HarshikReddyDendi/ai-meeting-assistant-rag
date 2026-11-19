from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader

from app.agents.action_item_extractor import ActionItem, ActionItemList
from app.agents.decision_extractor import DecisionList
from app.agents.summarizer import MeetingSummary
from app.transcription.models import TranscriptionResult
from app.utils.export_utils import (
    ExportError,
    export_action_items_pdf,
    export_action_items_txt,
    export_summary_pdf,
    export_summary_txt,
    export_transcript_txt,
)


def _sample_summary() -> MeetingSummary:
    return MeetingSummary(
        executive_summary="The team agreed on a Friday launch.",
        detailed_summary="The team reviewed readiness and launch timing.",
        main_topics=["Launch readiness"],
        unresolved_questions=["Final pricing"],
        follow_up_requirements=["Confirm support coverage"],
    )


def _sample_actions() -> ActionItemList:
    return ActionItemList(
        action_items=[
            ActionItem(task="Send release notes", owner="Mira", deadline="Friday"),
            ActionItem(task="Confirm support coverage"),
        ]
    )


def test_transcript_and_summary_text_exports_are_utf8() -> None:
    transcription = TranscriptionResult(
        text="नमस्ते बैठक",
        language="hi",
        segments=(),
    )

    transcript_bytes = export_transcript_txt(transcription)
    summary_bytes = export_summary_txt(
        _sample_summary(),
        decisions=DecisionList(decisions=["Ship on Friday"]),
        action_items=_sample_actions(),
    )

    assert transcript_bytes.decode("utf-8") == "नमस्ते बैठक"
    assert "Ship on Friday" in summary_bytes.decode("utf-8")
    assert "Owner: Not specified | Deadline: Not specified" in summary_bytes.decode(
        "utf-8"
    )


def test_action_items_text_export_includes_owner_and_deadline() -> None:
    text = export_action_items_txt(_sample_actions()).decode("utf-8")

    assert "Send release notes | Owner: Mira | Deadline: Friday" in text
    assert "Confirm support coverage | Owner: Not specified | Deadline: Not specified" in text


def test_summary_pdf_contains_summary_decisions_and_actions() -> None:
    pdf_bytes = export_summary_pdf(
        _sample_summary(),
        decisions=DecisionList(decisions=["Ship on Friday"]),
        action_items=_sample_actions(),
    )
    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(pdf_bytes)).pages
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert "Friday launch" in extracted
    assert "Ship on Friday" in extracted
    assert "Send release notes" in extracted
    assert "Not specified" in extracted


def test_action_items_pdf_uses_unicode_font_for_hindi_content() -> None:
    items = ActionItemList(action_items=[ActionItem(task="मसौदा भेजें")])

    pdf_bytes = export_action_items_pdf(items)
    extracted = "\n".join(
        page.extract_text() or "" for page in PdfReader(BytesIO(pdf_bytes)).pages
    )

    assert "मसौदा भेजें" in extracted


def test_non_latin_pdf_requires_an_existing_unicode_font(
    tmp_path: Path,
) -> None:
    summary = MeetingSummary(
        executive_summary="नमस्ते",
        detailed_summary="नमस्ते",
    )

    with pytest.raises(ExportError, match="font does not exist"):
        export_summary_pdf(summary, font_path=tmp_path / "missing.ttf")


def test_empty_transcript_cannot_be_exported() -> None:
    with pytest.raises(ValueError, match="transcript"):
        export_transcript_txt("  ")
