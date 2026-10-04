"""IntelliPDF - Hugging Face Spaces Entry Point.

This root-level app.py launches the full IntelliPDF application in single-process
mode for Hugging Face Spaces, routing all requests in-process through the RAG
pipeline and FastAPI service without requiring an external server or open ports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import runpy

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Force in-process mode for Hugging Face Spaces (single container / single process)
os.environ["INTELLIPDF_INPROCESS"] = "1"

# Initialize local SQLite database and data directories
from backend.config import ensure_directories
from backend.utils import db as db_module

ensure_directories()
db_module.init_db()

# Execute the complete Streamlit frontend in the current runtime context
frontend_entrypoint = ROOT_DIR / "frontend" / "app.py"
runpy.run_path(str(frontend_entrypoint), run_name="__main__")
