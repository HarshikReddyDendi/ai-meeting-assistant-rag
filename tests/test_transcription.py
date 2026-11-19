import wave
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest
import requests

from app import config
from app.transcription import audio_processor, sarvam_service, whisper_service
from app.transcription.models import TranscriptionError


def _write_test_wav(
    path: Path,
    *,
    channels: int = 2,
    sample_rate: int = 22050,
) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate)


def test_prepare_audio_creates_normalized_wav(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    output = tmp_path / "normalized.wav"
    _write_test_wav(source)

    result = audio_processor.prepare_audio_for_transcription(source, output)

    assert result == output.resolve()
    with wave.open(str(result), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000


def test_media_validation_rejects_unsupported_and_oversized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not media", encoding="utf-8")
    with pytest.raises(TranscriptionError, match="Unsupported"):
        audio_processor.validate_media_file(unsupported)

    source = tmp_path / "input.wav"
    _write_test_wav(source)
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_MB", 0)
    with pytest.raises(TranscriptionError, match="exceeds"):
        audio_processor.validate_media_file(source)


def test_ffmpeg_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.wav"
    _write_test_wav(source)

    def fail_ffmpeg(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> SimpleNamespace:
        assert command[0]
        assert not check
        assert capture_output
        assert text
        assert timeout == 600
        return SimpleNamespace(returncode=1, stderr="invalid media")

    monkeypatch.setattr(
        audio_processor.subprocess,
        "run",
        fail_ffmpeg,
    )

    with pytest.raises(TranscriptionError, match="invalid media"):
        audio_processor.prepare_audio_for_transcription(
            source,
            tmp_path / "output.wav",
        )


def test_whisper_transcription_preserves_language_and_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prepare_audio(media_path: str | Path, output_path: str | Path) -> Path:
        assert Path(media_path).suffix == ".wav"
        output = Path(output_path)
        _write_test_wav(output, channels=1, sample_rate=16000)
        return output

    class FakeWhisperModel:
        def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
            assert audio.size > 0
            assert kwargs["language"] == "hi"
            return {
                "text": "नमस्ते",
                "language": "hi",
                "segments": [{"start": 0.1, "end": 0.8, "text": " नमस्ते "}],
            }

    def load_fake_model(model_name: str) -> FakeWhisperModel:
        assert model_name == config.WHISPER_MODEL
        return FakeWhisperModel()

    monkeypatch.setattr(whisper_service, "prepare_audio_for_transcription", prepare_audio)
    monkeypatch.setattr(whisper_service, "_load_model", load_fake_model)
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"input")

    result = whisper_service.transcribe_audio(source, language="hi")

    assert result.text == "नमस्ते"
    assert result.language == "hi"
    assert result.segments[0].start == 0.1
    assert result.segments[0].speaker is None


def test_whisper_runtime_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prepare_audio(media_path: str | Path, output_path: str | Path) -> Path:
        assert Path(media_path).suffix == ".wav"
        output = Path(output_path)
        _write_test_wav(output, channels=1, sample_rate=16000)
        return output

    class FailedWhisperModel:
        def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
            assert audio.size > 0
            assert kwargs["verbose"] is False
            raise RuntimeError("simulated model failure")

    def load_failed_model(model_name: str) -> FailedWhisperModel:
        assert model_name == config.WHISPER_MODEL
        return FailedWhisperModel()

    monkeypatch.setattr(whisper_service, "prepare_audio_for_transcription", prepare_audio)
    monkeypatch.setattr(whisper_service, "_load_model", load_failed_model)
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"input")

    with pytest.raises(TranscriptionError, match="Whisper could not transcribe"):
        whisper_service.transcribe_audio(source)


def test_sarvam_transcription_parses_timestamped_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio bytes")
    monkeypatch.setattr(config, "SARVAM_API_KEY", "test-key")

    class FakeResponse:
        ok = True
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {
                "transcript": "Hello there",
                "language_code": "en-IN",
                "timestamps": {
                    "words": ["Hello there"],
                    "start_time_seconds": [0.0],
                    "end_time_seconds": [1.5],
                },
            }

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        assert url == sarvam_service.SARVAM_STT_URL
        assert kwargs["headers"]["api-subscription-key"] == "test-key"
        assert kwargs["data"]["language_code"] == "unknown"
        return FakeResponse()

    monkeypatch.setattr(sarvam_service.requests, "post", fake_post)

    result = sarvam_service.transcribe_with_sarvam(source)

    assert result.text == "Hello there"
    assert result.language == "en-IN"
    assert result.segments[0].end == 1.5
    assert result.segments[0].speaker is None


def test_sarvam_requires_api_key_without_network_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio bytes")
    monkeypatch.setattr(config, "SARVAM_API_KEY", "")

    with pytest.raises(TranscriptionError, match="SARVAM_API_KEY"):
        sarvam_service.transcribe_with_sarvam(source)


def test_sarvam_network_timeout_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio bytes")
    monkeypatch.setattr(config, "SARVAM_API_KEY", "test-key")

    def fake_post(url: str, **kwargs: Any) -> None:
        assert url == sarvam_service.SARVAM_STT_URL
        assert kwargs["timeout"] == (10, 120)
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(sarvam_service.requests, "post", fake_post)

    with pytest.raises(TranscriptionError, match="request failed") as exc_info:
        sarvam_service.transcribe_with_sarvam(source)

    assert isinstance(exc_info.value.__cause__, requests.Timeout)


def test_sarvam_quota_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio bytes")
    monkeypatch.setattr(config, "SARVAM_API_KEY", "test-key")

    class QuotaResponse:
        ok = False
        status_code = 429

    def fake_post(url: str, **kwargs: Any) -> QuotaResponse:
        assert url == sarvam_service.SARVAM_STT_URL
        assert kwargs["timeout"] == (10, 120)
        return QuotaResponse()

    monkeypatch.setattr(sarvam_service.requests, "post", fake_post)

    with pytest.raises(TranscriptionError, match="HTTP 429"):
        sarvam_service.transcribe_with_sarvam(source)
