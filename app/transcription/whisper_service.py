"""Local OpenAI Whisper transcription service."""

from __future__ import annotations

import logging
import tempfile
import wave
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app import config
from app.transcription.audio_processor import prepare_audio_for_transcription
from app.transcription.models import (
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> Any:
    """Load and cache the selected Whisper model."""
    import whisper

    return whisper.load_model(model_name)


def _read_normalized_audio(wav_path: Path) -> np.ndarray:
    """Read the normalized 16 kHz mono PCM WAV into Whisper's waveform format."""
    try:
        with wave.open(str(wav_path), "rb") as audio_file:
            if (
                audio_file.getnchannels() != 1
                or audio_file.getsampwidth() != 2
                or audio_file.getframerate() != 16000
            ):
                raise TranscriptionError(
                    "Normalized audio must be mono, 16 kHz, 16-bit PCM."
                )
            samples = audio_file.readframes(audio_file.getnframes())
    except (OSError, wave.Error) as exc:
        raise TranscriptionError("Could not read the normalized WAV audio.") from exc

    if not samples:
        raise TranscriptionError("The normalized WAV audio is empty.")
    return np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0


def _parse_whisper_result(result: Mapping[str, Any]) -> TranscriptionResult:
    text = result.get("text")
    language = result.get("language")
    raw_segments = result.get("segments", [])
    if not isinstance(text, str) or not isinstance(raw_segments, list):
        raise TranscriptionError("Whisper returned an invalid transcription result.")
    if language is not None and not isinstance(language, str):
        raise TranscriptionError("Whisper returned an invalid detected language.")

    segments: list[TranscriptSegment] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            raise TranscriptionError("Whisper returned an invalid transcript segment.")
        start = raw_segment.get("start")
        end = raw_segment.get("end")
        segment_text = raw_segment.get("text")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not isinstance(segment_text, str)
        ):
            raise TranscriptionError("Whisper returned an invalid transcript segment.")
        segments.append(
            TranscriptSegment(
                start=float(start),
                end=float(end),
                text=segment_text.strip(),
            )
        )

    return TranscriptionResult(
        text=text.strip(),
        language=language,
        segments=tuple(segments),
    )


def transcribe_audio(
    media_path: str | Path,
    *,
    model_name: str | None = None,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe supported audio/video locally with Whisper.

    Set ``language`` to a Whisper language code (for example ``"en"`` or
    ``"hi"``), or leave it unset for automatic detection.
    """
    selected_model = (model_name or config.WHISPER_MODEL).strip()
    if not selected_model:
        raise ValueError("model_name must not be empty.")

    try:
        with tempfile.TemporaryDirectory(prefix="meeting-audio-") as temp_dir:
            wav_path = prepare_audio_for_transcription(
                media_path,
                Path(temp_dir) / "normalized.wav",
            )
            waveform = _read_normalized_audio(wav_path)
            raw_result = _load_model(selected_model).transcribe(
                waveform,
                language=language,
                fp16=torch.cuda.is_available(),
                verbose=False,
            )
    except TranscriptionError:
        raise
    except (OSError, RuntimeError) as exc:
        logger.exception("Whisper transcription failed.")
        raise TranscriptionError("Whisper could not transcribe the media.") from exc

    if not isinstance(raw_result, Mapping):
        raise TranscriptionError("Whisper returned an invalid transcription result.")
    return _parse_whisper_result(raw_result)
