"""Shared pytest configuration for the IntelliPDF test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `backend`, `evaluation` and `frontend` importable when running
# `pytest` from anywhere inside the project.
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session", autouse=True)
def _quiet_third_party_logs() -> None:
    """Silence chatty third-party loggers during the test run."""
    import logging

    for name in ("httpx", "urllib3", "filelock", "datasets"):
        logging.getLogger(name).setLevel(logging.ERROR)