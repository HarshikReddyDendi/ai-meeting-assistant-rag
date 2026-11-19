"""Shared result types for transcription providers."""

from __future__ import annotations

from dataclasses import dataclass


class TranscriptionError(RuntimeError):
    """Raised when media cannot be transcribed or a provider fails."""


@dataclass(frozen=True)
class TranscriptSegment:
    """A transcript phrase and its optional time/speaker metadata."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Transcript text with detected language and available segments."""

    text: str
    language: str | None
    segments: tuple[TranscriptSegment, ...]
