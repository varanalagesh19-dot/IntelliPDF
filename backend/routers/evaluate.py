"""``POST /evaluate`` - grade a student's answer against the source."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import EvaluateRequest, EvaluateResponse, SourceRef
from backend.services.evaluator import evaluate_answer
from backend.utils import db

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    """Compare a student answer with the reference material.

    Returns the awarded marks, the points the student missed, any incorrect
    statements, improvement suggestions and the source pages used.
    """
    if not db.get_document(payload.doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document: {payload.doc_id}")

    try:
        result = evaluate_answer(
            doc_id=payload.doc_id,
            question=payload.question,
            student_answer=payload.student_answer,
            mark_type=payload.mark_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Evaluation failed for %s", payload.doc_id)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return EvaluateResponse(
        score_out_of=result["score_out_of"],
        score_earned=result["score_earned"],
        percentage=result["percentage"],
        missing_points=result["missing_points"],
        incorrect_points=result["incorrect_points"],
        suggestions=result["suggestions"],
        model_answer=result["model_answer"],
        sources=[SourceRef(**source) for source in result["sources"]],
    )