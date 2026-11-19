from types import SimpleNamespace
from collections.abc import Iterator

import pytest

from app.transcription import youtube_service
from app.transcription.models import TranscriptionError


@pytest.mark.parametrize(
    ("url", "expected_id"),
    [
        ("https://www.youtube.com/watch?v=abcdefghijk", "abcdefghijk"),
        ("https://youtu.be/abcdefghijk?t=42", "abcdefghijk"),
        ("https://youtube.com/shorts/abcdefghijk", "abcdefghijk"),
        ("https://m.youtube.com/live/abcdefghijk", "abcdefghijk"),
    ],
)
def test_extract_supported_youtube_video_ids(url: str, expected_id: str) -> None:
    assert youtube_service.extract_youtube_video_id(url) == expected_id


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com.evil.example/watch?v=abcdefghijk",
        "https://example.com/watch?v=abcdefghijk",
        "https://youtube.com/watch?v=short",
    ],
)
def test_reject_unsupported_youtube_urls(url: str) -> None:
    with pytest.raises(ValueError, match="YouTube"):
        youtube_service.extract_youtube_video_id(url)


def test_fetch_youtube_captions_preserves_segment_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTranscript:
        language_code = "en"

        def __iter__(self) -> Iterator[SimpleNamespace]:
            return iter(
                [
                    SimpleNamespace(text="Hello team", start=1.0, duration=2.0),
                    SimpleNamespace(text=" ", start=3.0, duration=1.0),
                ]
            )

    class FakeApi:
        def fetch(self, video_id: str, *, languages: list[str]) -> FakeTranscript:
            assert video_id == "abcdefghijk"
            assert languages == ["en", "hi"]
            return FakeTranscript()

    monkeypatch.setattr(youtube_service, "YouTubeTranscriptApi", FakeApi)

    result = youtube_service.fetch_youtube_captions(
        "https://youtu.be/abcdefghijk"
    )

    assert result.text == "Hello team"
    assert result.language == "en"
    assert result.segments[0].start == 1.0
    assert result.segments[0].end == 3.0


def test_unavailable_youtube_captions_return_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApi:
        def fetch(self, video_id: str, *, languages: list[str]) -> None:
            assert video_id == "abcdefghijk"
            assert languages == ["en", "hi"]
            raise youtube_service.YouTubeTranscriptApiException("captions disabled")

    monkeypatch.setattr(youtube_service, "YouTubeTranscriptApi", FakeApi)

    with pytest.raises(TranscriptionError, match="Captions are unavailable"):
        youtube_service.fetch_youtube_captions(
            "https://youtu.be/abcdefghijk"
        )
