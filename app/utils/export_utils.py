"""TXT and PDF report generation for meeting outputs."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from html import escape
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus.doctemplate import BaseDocTemplate
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

from app import config
from app.agents.action_item_extractor import ActionItemList
from app.agents.decision_extractor import DecisionList
from app.agents.summarizer import MeetingSummary
from app.transcription.models import TranscriptionResult

logger = logging.getLogger(__name__)

FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\Nirmala.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)


class ExportError(RuntimeError):
    """Raised when a requested export cannot be generated."""


def _transcript_text(transcript: str | TranscriptionResult) -> str:
    text = transcript.text if isinstance(transcript, TranscriptionResult) else transcript
    if not text.strip():
        raise ValueError("transcript must contain text.")
    return text


def export_transcript_txt(transcript: str | TranscriptionResult) -> bytes:
    """Return the full transcript as UTF-8 text bytes."""
    return _transcript_text(transcript).encode("utf-8")


def _summary_text(
    summary: MeetingSummary,
    decisions: DecisionList | None,
    action_items: ActionItemList | None,
) -> str:
    lines = [
        "MEETING SUMMARY",
        "",
        "Executive summary",
        summary.executive_summary,
        "",
        "Detailed summary",
        summary.detailed_summary,
        "",
        "Main topics",
        *(f"- {topic}" for topic in summary.main_topics),
        "",
        "Decisions",
        *(
            f"- {decision}"
            for decision in (decisions.decisions if decisions else [])
        ),
        "",
        "Action items",
        *(
            f"- {item.task} | Owner: {item.owner} | Deadline: {item.deadline}"
            for item in (action_items.action_items if action_items else [])
        ),
        "",
        "Unresolved questions",
        *(f"- {question}" for question in summary.unresolved_questions),
        "",
        "Follow-up requirements",
        *(f"- {requirement}" for requirement in summary.follow_up_requirements),
    ]
    return "\n".join(lines).strip() + "\n"


def export_summary_txt(
    summary: MeetingSummary,
    *,
    decisions: DecisionList | None = None,
    action_items: ActionItemList | None = None,
) -> bytes:
    """Return the summary, decisions, and action items as UTF-8 text bytes."""
    return _summary_text(summary, decisions, action_items).encode("utf-8")


def export_action_items_txt(action_items: ActionItemList) -> bytes:
    """Return action items with owners and deadlines as UTF-8 text bytes."""
    lines = ["ACTION ITEMS", ""]
    lines.extend(
        f"- {item.task} | Owner: {item.owner} | Deadline: {item.deadline}"
        for item in action_items.action_items
    )
    if not action_items.action_items:
        lines.append("No action items recorded.")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _font_name_for(text: str, font_path: str | Path | None) -> str:
    needs_unicode_font = any(ord(character) > 255 for character in text)
    configured_path = font_path or config.PDF_FONT_PATH
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if not candidate.is_file():
            raise ExportError(f"Configured PDF font does not exist: {candidate}")
    else:
        candidate = next((path for path in FONT_CANDIDATES if path.is_file()), None)
    if candidate is None:
        if needs_unicode_font:
            raise ExportError(
                "This PDF contains non-Latin text, but no Unicode font was found. "
                "Set PDF_FONT_PATH to a suitable .ttf or .ttc font."
            )
        return "Helvetica"

    font_name = "MeetingUnicode_" + hashlib.sha256(
        str(candidate.resolve()).encode("utf-8")
    ).hexdigest()[:10]
    try:
        if font_name not in pdfmetrics.getRegisteredFontNames():
            font = TTFont(font_name, str(candidate), subfontIndex=0, shapable=True)
            pdfmetrics.registerFont(font)
        font = pdfmetrics.getFont(font_name)
        missing_glyphs = {
            ord(character)
            for character in text
            if ord(character) > 255 and ord(character) not in font.face.charToGlyph
        }
        if missing_glyphs:
            raise ExportError(
                f"The configured PDF font does not support all characters "
                f"(for example U+{min(missing_glyphs):04X})."
            )
    except (OSError, TTFError) as exc:
        logger.exception("Could not load PDF font %s.", candidate)
        raise ExportError(f"Could not load PDF font: {candidate}") from exc
    return font_name


def _paragraph_text(text: str) -> str:
    return escape(text).replace("\n", "<br/>")


def _styles(font_name: str) -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "MeetingTitle",
            fontName=font_name,
            fontSize=19,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "heading": ParagraphStyle(
            "MeetingHeading",
            fontName=font_name,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#183B56"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "MeetingBody",
            fontName=font_name,
            fontSize=10,
            leading=15,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "MeetingTable",
            fontName=font_name,
            fontSize=9,
            leading=12,
            wordWrap="CJK",
        ),
    }


def _section(
    story: list[Flowable],
    title: str,
    content: list[str],
    styles: dict[str, ParagraphStyle],
) -> None:
    story.append(Paragraph(_paragraph_text(title), styles["heading"]))
    if content:
        story.extend(
            Paragraph(_paragraph_text(item), styles["body"])
            for item in content
        )
    else:
        story.append(Paragraph("None recorded.", styles["body"]))


def _action_table(
    action_items: ActionItemList | None,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    if action_items is None or not action_items.action_items:
        return [Paragraph("No action items recorded.", styles["body"])]

    data = [
        [
            Paragraph("Task", styles["table"]),
            Paragraph("Owner", styles["table"]),
            Paragraph("Deadline", styles["table"]),
        ]
    ]
    data.extend(
        [
            Paragraph(_paragraph_text(item.task), styles["table"]),
            Paragraph(_paragraph_text(item.owner), styles["table"]),
            Paragraph(_paragraph_text(item.deadline), styles["table"]),
        ]
        for item in action_items.action_items
    )
    table = Table(data, colWidths=[3.5 * inch, 1.5 * inch, 1.5 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF2")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C4CC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return [table]


def _build_pdf(
    title: str,
    content_text: str,
    story_builder: Callable[[dict[str, ParagraphStyle]], list[Flowable]],
    *,
    font_path: str | Path | None = None,
) -> bytes:
    font_name = _font_name_for(content_text, font_path)
    styles = _styles(font_name)
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=title,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
    )

    def draw_page_number(canvas: Canvas, doc: BaseDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawString(doc.leftMargin, 0.35 * inch, title)
        canvas.drawRightString(
            A4[0] - doc.rightMargin,
            0.35 * inch,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    story = story_builder(styles)
    document.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    return buffer.getvalue()


def export_summary_pdf(
    summary: MeetingSummary,
    *,
    decisions: DecisionList | None = None,
    action_items: ActionItemList | None = None,
    font_path: str | Path | None = None,
) -> bytes:
    """Generate a PDF meeting summary including decisions and action items."""
    plain_text = _summary_text(summary, decisions, action_items)

    def build_story(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
        story: list[Flowable] = [
            Paragraph("Meeting summary", styles["title"]),
        ]
        _section(story, "Executive summary", [summary.executive_summary], styles)
        _section(story, "Detailed summary", [summary.detailed_summary], styles)
        _section(story, "Main topics", summary.main_topics, styles)
        _section(
            story,
            "Decisions",
            decisions.decisions if decisions else [],
            styles,
        )
        story.append(Paragraph("Action items", styles["heading"]))
        story.extend(_action_table(action_items, styles))
        _section(
            story,
            "Unresolved questions",
            summary.unresolved_questions,
            styles,
        )
        _section(
            story,
            "Follow-up requirements",
            summary.follow_up_requirements,
            styles,
        )
        return story

    return _build_pdf(
        "Meeting summary",
        plain_text,
        build_story,
        font_path=font_path,
    )


def export_action_items_pdf(
    action_items: ActionItemList,
    *,
    font_path: str | Path | None = None,
) -> bytes:
    """Generate a PDF action-item report."""
    plain_text = export_action_items_txt(action_items).decode("utf-8")

    def build_story(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
        return [
            Paragraph("Action items", styles["title"]),
            *_action_table(action_items, styles),
        ]

    return _build_pdf(
        "Action items",
        plain_text,
        build_story,
        font_path=font_path,
    )
