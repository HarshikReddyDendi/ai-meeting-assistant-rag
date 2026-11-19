from pathlib import Path

import pytest

from app import config
from app.utils.file_handler import save_transcript, save_uploaded_media


def test_upload_uses_generated_path_not_supplied_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)

    stored = save_uploaded_media(r"..\..\private\recording.MP4", b"video data")

    assert stored.original_name == "recording.MP4"
    assert stored.path.parent == tmp_path
    assert stored.path.name == f"{stored.meeting_id}.mp4"
    assert stored.path.read_bytes() == b"video data"


def test_upload_rejects_unsupported_and_oversized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        save_uploaded_media("notes.txt", b"content")

    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_MB", 0)
    with pytest.raises(ValueError, match="exceeds"):
        save_uploaded_media("meeting.wav", b"audio")


def test_transcript_storage_rejects_unsafe_meeting_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)

    with pytest.raises(ValueError, match="meeting_id"):
        save_transcript("..\\outside", "Transcript text.")
