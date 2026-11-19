"""Validate media files and convert audio/video to normalized WAV."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import imageio_ffmpeg

from app import config
from app.transcription.models import TranscriptionError

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".amr",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
VIDEO_EXTENSIONS = {
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
SUPPORTED_MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def validate_media_file(media_path: str | Path) -> Path:
    """Validate a supported local media file and configured size limit."""
    path = Path(media_path).expanduser()
    if not path.is_file():
        raise TranscriptionError(f"Media file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        raise TranscriptionError(
            f"Unsupported media format '{path.suffix or '(none)'}'."
        )
    max_size = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if path.stat().st_size == 0:
        raise TranscriptionError("Media file is empty.")
    if path.stat().st_size > max_size:
        raise TranscriptionError(
            f"Media file exceeds the {config.MAX_UPLOAD_SIZE_MB} MB limit."
        )
    return path.resolve()


def prepare_audio_for_transcription(
    media_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Convert supported audio/video to mono 16 kHz PCM WAV using FFmpeg."""
    source = validate_media_file(media_path)
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise TranscriptionError("The WAV output path must differ from the input.")
    if destination.suffix.lower() != ".wav":
        raise TranscriptionError("The normalized audio output must use .wav.")
    if destination.exists():
        raise TranscriptionError(f"Audio output already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:
        raise TranscriptionError("The bundled FFmpeg executable is unavailable.") from exc

    command = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError("FFmpeg timed out while processing the media.") from exc
    except OSError as exc:
        logger.exception("Could not start FFmpeg for media conversion.")
        raise TranscriptionError("Could not start FFmpeg for media conversion.") from exc

    if result.returncode != 0:
        details = result.stderr.strip()[-1000:]
        logger.warning("FFmpeg media conversion failed: %s", details)
        raise TranscriptionError(
            f"FFmpeg could not process the media file. {details}"
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise TranscriptionError("FFmpeg did not produce a valid WAV file.")
    return destination


def extract_audio_from_video(
    video_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Extract and normalize audio from a supported video file."""
    source = validate_media_file(video_path)
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise TranscriptionError("The input file is not a supported video format.")
    return prepare_audio_for_transcription(source, output_path)
