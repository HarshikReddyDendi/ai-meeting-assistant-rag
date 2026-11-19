"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(value: str) -> Path:
    """Resolve a configured path relative to the project root when needed."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "saaras:v4")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
CHROMA_DB_PATH = _project_path(os.getenv("CHROMA_DB_PATH", "data/vectorstore"))
UPLOAD_DIR = _project_path(os.getenv("UPLOAD_DIR", "data/uploads"))
EXPORT_DIR = _project_path(os.getenv("EXPORT_DIR", "exports"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
PDF_FONT_PATH = os.getenv("PDF_FONT_PATH", "")


def ensure_runtime_directories() -> None:
    """Create directories used for local runtime data."""
    for directory in (CHROMA_DB_PATH, UPLOAD_DIR, EXPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)