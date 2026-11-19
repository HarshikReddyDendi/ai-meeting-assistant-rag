"""Safe storage helpers for uploaded media and generated transcripts."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app import config
from app.transcription.audio_processor import SUPPORTED_MEDIA_EXTENSIONS

logger = logging.getLogger(__name__)
MEETING_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class FileStorageError(RuntimeError):
    """Raised when an upload or transcript cannot be stored."""


@dataclass(frozen=True)
class StoredMedia:
    """A safely named uploaded recording."""

    meeting_id: str
    original_name: str
    path: Path


def create_meeting_id() -> str:
    """Create a filesystem-safe identifier for a new meeting."""
    return uuid.uuid4().hex


def save_uploaded_media(file_name: str, content: bytes) -> StoredMedia:
    """Validate and store media under a generated ID, never its user filename."""
    original_name = file_name.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_MEDIA_EXTENSIONS:
        raise ValueError(f"Unsupported media format '{extension or '(none)'}'.")
    if not content:
        raise ValueError("The uploaded media file is empty.")
    max_size = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_size:
        raise ValueError(
            f"The uploaded file exceeds the {config.MAX_UPLOAD_SIZE_MB} MB limit."
        )

    meeting_id = create_meeting_id()
    destination = config.UPLOAD_DIR / f"{meeting_id}{extension}"
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as file:
            file.write(content)
    except OSError as exc:
        logger.exception("Could not store uploaded meeting media.")
        raise FileStorageError("Could not store the uploaded media file.") from exc
    return StoredMedia(
        meeting_id=meeting_id,
        original_name=original_name,
        path=destination,
    )


def save_transcript(meeting_id: str, transcript: str) -> Path:
    """Persist transcript text using a validated generated meeting ID."""
    if not MEETING_ID_PATTERN.fullmatch(meeting_id):
        raise ValueError("meeting_id must be a generated 32-character ID.")
    if not transcript.strip():
        raise ValueError("transcript must contain text.")

    destination = config.UPLOAD_DIR / f"{meeting_id}.txt"
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(transcript, encoding="utf-8")
    except OSError as exc:
        logger.exception("Could not save transcript for meeting %s.", meeting_id)
        raise FileStorageError("Could not save the transcript.") from exc
    return destination
