"""Load and normalize transcript text into LangChain documents."""

from __future__ import annotations

import math
import re
from pathlib import Path

from langchain_core.documents import Document

from app.transcription.models import TranscriptionResult


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _format_timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Transcript timestamps must be finite and non-negative.")
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def load_transcript(
    transcript: str | Path | TranscriptionResult,
    *,
    meeting_id: str,
) -> list[Document]:
    """Load a text transcript or transcription result with meeting metadata."""
    meeting_id = meeting_id.strip()
    if not meeting_id:
        raise ValueError("meeting_id must not be empty.")

    if isinstance(transcript, TranscriptionResult):
        if transcript.segments:
            documents: list[Document] = []
            for segment in transcript.segments:
                if segment.end < segment.start:
                    raise ValueError("Transcript segment end precedes its start.")
                content = _normalize_text(
                    f"[{_format_timestamp(segment.start)}] {segment.text}"
                )
                if not content:
                    continue
                metadata: dict[str, str | float] = {
                    "meeting_id": meeting_id,
                    "source": "transcript",
                    "start": segment.start,
                    "end": segment.end,
                }
                if segment.speaker is not None:
                    metadata["speaker"] = segment.speaker
                documents.append(Document(page_content=content, metadata=metadata))
            if documents:
                return documents
        text = transcript.text
        source = "transcript"
    elif isinstance(transcript, Path):
        source = transcript.name
        try:
            text = transcript.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read transcript file: {transcript}") from exc
    else:
        text = transcript
        source = "transcript"

    normalized_text = _normalize_text(text)
    if not normalized_text:
        raise ValueError("transcript must contain text.")
    return [
        Document(
            page_content=normalized_text,
            metadata={"meeting_id": meeting_id, "source": source},
        )
    ]
