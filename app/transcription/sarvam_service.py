"""Optional Sarvam AI speech-to-text integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import requests

from app import config
from app.transcription.audio_processor import validate_media_file
from app.transcription.models import (
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SUPPORTED_LANGUAGE_CODES = {
    "as-IN",
    "bn-IN",
    "brx-IN",
    "doi-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "kok-IN",
    "ks-IN",
    "mai-IN",
    "ml-IN",
    "mni-IN",
    "mr-IN",
    "ne-IN",
    "od-IN",
    "pa-IN",
    "sa-IN",
    "sat-IN",
    "sd-IN",
    "ta-IN",
    "te-IN",
    "ur-IN",
}


def transcribe_with_sarvam(
    media_path: str | Path,
    *,
    language_code: str = "unknown",
    model: str | None = None,
) -> TranscriptionResult:
    """Transcribe a short supported media clip using Sarvam's REST API.

    Sarvam's synchronous REST endpoint is documented for quick responses under
    30 seconds; use local Whisper for full-length meeting recordings.
    """
    api_key = config.SARVAM_API_KEY.strip()
    if not api_key:
        raise TranscriptionError(
            "SARVAM_API_KEY is not configured. Add it to the project .env file."
        )
    if language_code != "unknown" and language_code not in SUPPORTED_LANGUAGE_CODES:
        raise ValueError(f"Unsupported Sarvam language code: {language_code}")

    selected_model = (model or config.SARVAM_MODEL).strip()
    if not selected_model:
        raise ValueError("model must not be empty.")
    source = validate_media_file(media_path)
    form_data = {
        "model": selected_model,
        "language_code": language_code,
        "mode": "transcribe",
        "with_timestamps": "true",
    }

    try:
        with source.open("rb") as audio_file:
            response = requests.post(
                SARVAM_STT_URL,
                headers={"api-subscription-key": api_key},
                data=form_data,
                files={"file": (source.name, audio_file)},
                timeout=(10, 120),
            )
    except requests.RequestException as exc:
        logger.exception("Sarvam transcription request failed.")
        raise TranscriptionError("Sarvam transcription request failed.") from exc

    if not response.ok:
        logger.warning("Sarvam transcription returned HTTP %s.", response.status_code)
        raise TranscriptionError(
            f"Sarvam transcription failed with HTTP {response.status_code}."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise TranscriptionError("Sarvam returned an invalid JSON response.") from exc
    if not isinstance(payload, Mapping):
        raise TranscriptionError("Sarvam returned an invalid response.")

    transcript = payload.get("transcript")
    detected_language = payload.get("language_code")
    if not isinstance(transcript, str):
        raise TranscriptionError("Sarvam response did not contain transcript text.")
    if detected_language is not None and not isinstance(detected_language, str):
        raise TranscriptionError("Sarvam returned an invalid detected language.")

    return TranscriptionResult(
        text=transcript.strip(),
        language=detected_language,
        segments=_parse_timestamps(payload.get("timestamps")),
    )


def _parse_timestamps(raw_timestamps: object) -> tuple[TranscriptSegment, ...]:
    if raw_timestamps is None:
        return ()
    if not isinstance(raw_timestamps, Mapping):
        raise TranscriptionError("Sarvam returned invalid transcript timestamps.")

    words = raw_timestamps.get("words")
    starts = raw_timestamps.get("start_time_seconds")
    ends = raw_timestamps.get("end_time_seconds")
    if (
        not isinstance(words, list)
        or not isinstance(starts, list)
        or not isinstance(ends, list)
        or len(words) != len(starts)
        or len(words) != len(ends)
    ):
        raise TranscriptionError("Sarvam returned invalid transcript timestamps.")

    segments: list[TranscriptSegment] = []
    for text, start, end in zip(words, starts, ends):
        if (
            not isinstance(text, str)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise TranscriptionError("Sarvam returned invalid transcript timestamps.")
        segments.append(
            TranscriptSegment(start=float(start), end=float(end), text=text)
        )
    return tuple(segments)
