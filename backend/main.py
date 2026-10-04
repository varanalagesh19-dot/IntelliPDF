"""FastAPI application entry point for IntelliPDF.

Run with::

    uvicorn backend.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the interactive API.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    APP_NAME,
    DISCLAIMER,
    EMBEDDING_MODEL,
    INSUFFICIENT_INFO_ANSWER,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_UPLOAD_MB,
    OLLAMA_MODEL,
    SIMILARITY_THRESHOLD,
    TAGLINE,
    TOP_K,
    ensure_directories,
    setup_logging,
)
from backend.routers import chat as chat_router
from backend.routers import evaluate as evaluate_router
from backend.routers import mock_exam as mock_exam_router
from backend.routers import quiz as quiz_router
from backend.routers import study_plan as study_plan_router
from backend.routers import upload as upload_router
from backend.services.embedder import get_embedder
from backend.services.llm_client import get_llm_client
from backend.services.vector_store import get_vector_store
from backend.utils import db
from backend.utils.language import LANG_INSTRUCTIONS, SUPPORTED_LANGUAGES
from backend.services.mark_wise import MARK_TYPES

LOGGER = logging.getLogger(__name__)

VERSION = "1.0.0"

DESCRIPTION = f"""
**{APP_NAME}** - {TAGLINE}

Upload any subject PDF → Understand → Retrieve → Generate.

* Foundation: NLP → Text Embeddings → Vector Search → RAG → LLMs
* Everything runs locally: FAISS + SQLite + local files, free-tier LLMs only
* Evaluation: SQuAD 2.0 (EM / F1) via `evaluation/squad_eval.py`

_{DISCLAIMER}_
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare directories, the database and the embedding model on startup."""
    setup_logging()
    ensure_directories()
    db.init_db()
    LOGGER.info("Starting %s v%s", APP_NAME, VERSION)

    try:  # Loading MiniLM takes ~10s the first time; never block startup on it.
        get_embedder().load()
        LOGGER.info("Embedder ready: %s", get_embedder().model_name)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Embedder warm-up failed (%s); lazy loading will retry.", exc)

    yield
    LOGGER.info("Shutting down %s", APP_NAME)


app = FastAPI(
    title=f"{APP_NAME} API",
    description=DESCRIPTION,
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo: Streamlit on :8501, API on :8000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next: Any) -> JSONResponse:
    """Log every request with its duration (useful when demoing)."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    LOGGER.info("%s %s -> %s (%.0f ms)", request.method, request.url.path,
                response.status_code, elapsed)
    return response


# ── Routers ──────────────────────────────────────────────────────────────
app.include_router(upload_router.router)
app.include_router(chat_router.router)
app.include_router(quiz_router.router)
app.include_router(evaluate_router.router)
app.include_router(study_plan_router.router)
app.include_router(mock_exam_router.router)


# ── Core endpoints ───────────────────────────────────────────────────────


@app.get("/health", tags=["core"])
def health() -> dict[str, Any]:
    """Liveness probe plus a per-service availability report."""
    llm_health = get_llm_client().health()
    embedder_loaded = get_embedder().model_name != "not-loaded"
    try:
        documents = db.list_documents(limit=1000)
        db_ok = True
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Database health check failed: %s", exc)
        documents = []
        db_ok = False

    try:
        get_vector_store().base_path.mkdir(parents=True, exist_ok=True)
        vector_ok = True
    except Exception:  # noqa: BLE001
        vector_ok = False

    return {
        "status": "ok",
        "version": VERSION,
        "tagline": TAGLINE,
        "disclaimer": DISCLAIMER,
        "services": {
            "llm": llm_health,
            "embedder": {
                "model": get_embedder().model_name,
                "configured_model": EMBEDDING_MODEL,
                "loaded": embedder_loaded,
                "dim": get_embedder().dim,
            },
            "vector_store": {
                "path": str(get_vector_store().base_path),
                "writable": vector_ok,
                "documents": len(documents),
            },
            "database": {"ok": db_ok, "documents": len(documents)},
            "rag": {
                "top_k": TOP_K,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "abstention_message": INSUFFICIENT_INFO_ANSWER,
            },
            "features": {
                "mark_types": list(MARK_TYPES),
                "languages": list(SUPPORTED_LANGUAGES),
                "language_instructions": LANG_INSTRUCTIONS,
            },
            "llm_settings": {
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
                "ollama_model": OLLAMA_MODEL,
            },
            "limits": {"max_upload_mb": MAX_UPLOAD_MB},
        },
    }


@app.get("/", tags=["core"])
def root() -> dict[str, str]:
    """Friendly entry point with the exact commands to get started."""
    return {
        "app": APP_NAME,
        "tagline": TAGLINE,
        "docs": "/docs",
        "health": "/health",
        "frontend": "streamlit run frontend/app.py --server.port 8501",
        "upload": "POST /upload (multipart: file=@notes.pdf, syllabus=optional)",
        "disclaimer": DISCLAIMER,
    }