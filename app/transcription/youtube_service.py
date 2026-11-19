"""Retrieve available captions from supported YouTube URLs."""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi, YouTubeTranscriptApiException

from app.transcription.models import (
    TranscriptSegment,
    TranscriptionError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_youtube_video_id(url: str) -> str:
    """Validate a supported YouTube URL and return its video ID."""
    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or hostname not in YOUTUBE_HOSTS:
        raise ValueError("Enter a valid YouTube video URL.")

    video_id = ""
    if hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/", maxsplit=1)[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) == 2 and path_parts[0] in {"shorts", "embed", "live"}:
            video_id = path_parts[1]

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("The YouTube URL does not contain a valid video ID.")
    return video_id


def fetch_youtube_captions(url: str) -> TranscriptionResult:
    """Fetch English or Hindi captions without downloading the video."""
    video_id = extract_youtube_video_id(url)
    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en", "hi"],
        )
    except YouTubeTranscriptApiException as exc:
        logger.exception("Could not retrieve YouTube captions for video %s.", video_id)
        raise TranscriptionError(
            "Captions are unavailable for this video, or YouTube could not provide them."
        ) from exc

    segments = tuple(
        TranscriptSegment(
            start=float(snippet.start),
            end=float(snippet.start + snippet.duration),
            text=snippet.text,
        )
        for snippet in transcript
        if snippet.text.strip()
    )
    text = "\n".join(segment.text for segment in segments)
    if not text.strip():
        raise TranscriptionError("The YouTube captions were empty.")
    return TranscriptionResult(
        text=text,
        language=transcript.language_code,
        segments=segments,
    )
