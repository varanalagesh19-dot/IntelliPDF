"""Central configuration for IntelliPDF.

Loads environment variables (``.env``) once and exposes resolved absolute
paths plus the tuning constants shared by the services.  Everything here is
local-first: no hosted database, no paid API.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────
# backend/config.py -> backend/ -> intellipdf/
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BACKEND_DIR: Final[Path] = PROJECT_ROOT / "backend"
DATA_DIR: Final[Path] = BACKEND_DIR / "data"

# .env is looked up in the project root (and next to this file as a fallback)
for _candidate in (PROJECT_ROOT / ".env", BACKEND_DIR / ".env"):
    if _candidate.exists():
        load_dotenv(_candidate, override=False)

# ── Environment helpers ──────────────────────────────────────────────────


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value.strip() if value and value.strip() else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid integer for %s, falling back to %s", key, default
        )
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid float for %s, falling back to %s", key, default
        )
        return default


def resolve_path(value: str, default: Path) -> Path:
    """Resolve a configured path relative to the project root when needed.

    Args:
        value: Raw path string taken from the environment.
        default: Project-relative default used when the value is empty.

    Returns:
        An absolute, existing-directory :class:`~pathlib.Path`.
    """
    raw = value or str(default)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


# ── Storage ──────────────────────────────────────────────────────────────
VECTOR_DB_PATH: Final[Path] = resolve_path(
    _env_str("VECTOR_DB_PATH", "./backend/data/vector_db"),
    DATA_DIR / "vector_db",
)
UPLOAD_PATH: Final[Path] = resolve_path(
    _env_str("UPLOAD_PATH", "./backend/data/uploads"), DATA_DIR / "uploads"
)
DB_PATH: Final[Path] = resolve_path(
    _env_str("DB_PATH", "./backend/data/intellipdf.db"), DATA_DIR / "intellipdf.db"
)

# ── LLM providers ────────────────────────────────────────────────────────
GROQ_API_KEY: Final[str] = _env_str("GROQ_API_KEY", "")
GROQ_MODEL: Final[str] = _env_str("GROQ_MODEL", "llama-3.1-70b-versatile")
GEMINI_API_KEY: Final[str] = _env_str("GEMINI_API_KEY", "")
GEMINI_MODEL: Final[str] = _env_str("GEMINI_MODEL", "gemini-1.5-flash")
OLLAMA_URL: Final[str] = _env_str("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: Final[str] = _env_str("OLLAMA_MODEL", "llama3")
DEFAULT_LLM: Final[str] = _env_str("DEFAULT_LLM", "groq").lower()
LLM_TEMPERATURE: Final[float] = _env_float("LLM_TEMPERATURE", 0.3)
LLM_MAX_TOKENS: Final[int] = _env_int("LLM_MAX_TOKENS", 1500)
LLM_TIMEOUT_SECONDS: Final[int] = _env_int("LLM_TIMEOUT_SECONDS", 90)

# ── Retrieval / chunking ─────────────────────────────────────────────────
EMBEDDING_MODEL: Final[str] = _env_str(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
#: Which embedder to use: ``auto`` (sentence-transformers, fall back on error),
#: ``sentence-transformers`` (force MiniLM) or ``hashing`` (no torch, ~50 MB RAM
#: instead of ~1 GB - the safe choice on 1 GB hosts such as Streamlit Cloud).
EMBEDDING_BACKEND: Final[str] = _env_str("EMBEDDING_BACKEND", "auto").lower()
CHUNK_SIZE: Final[int] = _env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP: Final[int] = _env_int("CHUNK_OVERLAP", 150)
TOP_K: Final[int] = _env_int("TOP_K", 5)
SIMILARITY_THRESHOLD: Final[float] = _env_float("SIMILARITY_THRESHOLD", 0.12)
EMBED_BATCH_SIZE: Final[int] = _env_int("EMBED_BATCH_SIZE", 32)
#: Hybrid retrieval mix: dense (MiniLM cosine) + IDF-weighted lexical overlap.
HYBRID_DENSE_WEIGHT: Final[float] = _env_float("HYBRID_DENSE_WEIGHT", 0.6)
HYBRID_LEXICAL_WEIGHT: Final[float] = _env_float("HYBRID_LEXICAL_WEIGHT", 0.4)

# ── Upload limits ────────────────────────────────────────────────────────
MAX_UPLOAD_MB: Final[int] = _env_int("MAX_UPLOAD_MB", 50)

# ── Firebase Authentication ──────────────────────────────────────────────
FIREBASE_API_KEY: Final[str] = _env_str("FIREBASE_API_KEY", "")
FIREBASE_AUTH_DOMAIN: Final[str] = _env_str("FIREBASE_AUTH_DOMAIN", "")
FIREBASE_PROJECT_ID: Final[str] = _env_str("FIREBASE_PROJECT_ID", "")
FIREBASE_STORAGE_BUCKET: Final[str] = _env_str("FIREBASE_STORAGE_BUCKET", "")
FIREBASE_MESSAGING_SENDER_ID: Final[str] = _env_str("FIREBASE_MESSAGING_SENDER_ID", "")
FIREBASE_APP_ID: Final[str] = _env_str("FIREBASE_APP_ID", "")
FIREBASE_SERVICE_ACCOUNT_PATH: Final[Path] = resolve_path(
    _env_str("FIREBASE_SERVICE_ACCOUNT_PATH", "./firebase-service-account.json"),
    PROJECT_ROOT / "firebase-service-account.json",
)
ALLOWED_PDF_SUFFIXES: Final[tuple[str, ...]] = (".pdf",)

# ── Server ───────────────────────────────────────────────────────────────
API_HOST: Final[str] = _env_str("API_HOST", "127.0.0.1")
API_PORT: Final[int] = _env_int("API_PORT", 8000)
LOG_LEVEL: Final[str] = _env_str("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Configure root logging once, at INFO level, with timestamps."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Third-party chatter is not useful for a demo.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_directories() -> None:
    """Create the upload / vector-store / data directories if missing."""
    for directory in (UPLOAD_PATH, VECTOR_DB_PATH, DB_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)


# The answer returned when retrieval confidence is too low to answer.
INSUFFICIENT_INFO_ANSWER: Final[str] = (
    "Insufficient information in the uploaded material."
)

APP_NAME: Final[str] = "IntelliPDF"
TAGLINE: Final[str] = "AI-Powered Personalized Learning and Exam Preparation System"
DISCLAIMER: Final[str] = (
    "Identifies topics for focused study; does not predict the actual "
    "question paper."
)