"""``POST /ask`` - mark-wise, source-grounded question answering (RAG)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from backend.config import TOP_K
from backend.models.schemas import AskRequest, AskResponse
from backend.services.mark_wise import MARK_TYPES, marks_distribution
from backend.services.rag_pipeline import rag_query, retrieve
from backend.services.vector_store import get_vector_store
from backend.utils import db
from backend.utils.language import LANG_INSTRUCTIONS, SUPPORTED_LANGUAGES

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _require_doc(doc_id: str) -> None:
    """Raise 404 when the document or its vector index is missing."""
    if not db.get_document(doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    if not get_vector_store().exists(doc_id):
        raise HTTPException(
            status_code=404,
            detail=f"No vector index for {doc_id}. Re-upload the PDF.",
        )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """Answer a question about an uploaded PDF.

    Retrieval is capped at top-5 chunks; the answer is rewritten to the
    requested mark count and language, and every source page is returned so the
    student can verify it.
    """
    _require_doc(payload.doc_id)
    started = time.perf_counter()

    try:
        result = rag_query(
            doc_id=payload.doc_id,
            query=payload.question,
            mark_type=payload.mark_type,
            language=payload.language,
            k=TOP_K,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("RAG query failed for %s", payload.doc_id)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        confidence_label=result["confidence_label"],
        provider=result["provider"],
        model=result.get("model", ""),
        mark_type=result["mark_type"],
        language=result["language"],
        grounded=result["grounded"],
        elapsed_ms=elapsed_ms,
    )


@router.get("/ask/capabilities")
def capabilities() -> dict[str, object]:
    """Expose mark types and languages so the UI stays in sync with the API."""
    return {
        "mark_types": MARK_TYPES,
        "mark_guidance": {mark: marks_distribution(mark) for mark in MARK_TYPES},
        "languages": list(SUPPORTED_LANGUAGES),
        "language_instructions": LANG_INSTRUCTIONS,
    }


@router.get("/ask/preview/{doc_id}")
def preview_sources(doc_id: str, query: str = "summary") -> dict[str, object]:
    """Return the top retrieved chunks without calling the LLM (debug aid)."""
    _require_doc(doc_id)
    results = retrieve(doc_id, query, k=TOP_K)
    return {
        "query": query,
        "results": [
            {
                "chunk_id": result["chunk"]["chunk_id"],
                "page": result["chunk"]["page_num"],
                "score": round(result["score"], 4),
                "text": result["chunk"]["text"][:400],
            }
            for result in results
        ],
    }