"""``POST /study-plan`` and ``POST /revision``.

The study planner turns an exam date + available hours into a weighted,
day-wise schedule (weak topics get 1.5x the time).  The revision endpoint
produces flashcards for last-minute revision, definitions or key concepts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    Flashcard,
    RevisionRequest,
    RevisionResponse,
    SourceRef,
    StudyPlanDay,
    StudyPlanRequest,
    StudyPlanResponse,
)
from backend.services.revision import generate_revision_cards
from backend.services.study_planner import generate_study_plan
from backend.utils import db

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["study"])


@router.post("/study-plan", response_model=StudyPlanResponse)
def create_study_plan(payload: StudyPlanRequest) -> StudyPlanResponse:
    """Generate a day-wise study plan for an upcoming exam."""
    if not db.get_document(payload.doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document: {payload.doc_id}")

    try:
        plan = generate_study_plan(
            doc_id=payload.doc_id,
            exam_date=payload.exam_date,
            daily_hours=payload.daily_hours,
            weak_topics=payload.weak_topics or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Study plan generation failed for %s", payload.doc_id)
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {exc}") from exc

    return StudyPlanResponse(
        doc_id=plan["doc_id"],
        exam_date=plan["exam_date"],
        days_left=plan["days_left"],
        daily_hours=plan["daily_hours"],
        total_hours=plan["total_hours"],
        topics=plan["topics"],
        plan=[StudyPlanDay(**day) for day in plan["plan"]],
        disclaimer=plan["disclaimer"],
    )


@router.get("/study-plan/{doc_id}/latest")
def latest_study_plan(doc_id: str) -> dict[str, object]:
    """Return the most recently generated study plan for a document."""
    plan = db.latest_study_plan(doc_id)
    if not plan:
        raise HTTPException(status_code=404, detail="No study plan generated yet.")
    return plan


@router.post("/revision", response_model=RevisionResponse)
def revision(payload: RevisionRequest) -> RevisionResponse:
    """Generate revision flashcards from an uploaded PDF."""
    if not db.get_document(payload.doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document: {payload.doc_id}")

    try:
        result = generate_revision_cards(
            doc_id=payload.doc_id,
            mode=payload.mode,
            num_cards=payload.num_cards,
            language=payload.language,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Revision generation failed for %s", payload.doc_id)
        raise HTTPException(status_code=500, detail=f"Revision failed: {exc}") from exc

    return RevisionResponse(
        doc_id=payload.doc_id,
        mode=result["mode"],
        cards=[Flashcard(**card) for card in result["cards"]],
        sources=[SourceRef(**source) for source in result["sources"]],
    )