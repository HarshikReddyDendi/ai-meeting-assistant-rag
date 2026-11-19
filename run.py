"""Run the Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    app_path = Path(__file__).parent / "app" / "main.py"
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])
    )
