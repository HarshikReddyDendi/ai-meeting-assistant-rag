from pathlib import Path

from app import config


def test_runtime_paths_are_under_project_root() -> None:
    """Configured default runtime paths resolve inside this project."""
    for path in (config.CHROMA_DB_PATH, config.UPLOAD_DIR, config.EXPORT_DIR):
        assert isinstance(path, Path)
        assert config.PROJECT_ROOT in path.parents
