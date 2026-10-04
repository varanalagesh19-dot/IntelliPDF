"""``POST /quiz``, ``POST /quiz/submit`` and ``GET /weak-topics/{doc_id}``."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    QuestionFeedback,
    QuizGenerateRequest,
    QuizGenerateResponse,
    QuizQuestion,
    QuizSubmitRequest,
    QuizSubmitResponse,
    TopicAccuracy,
    WeakTopicsResponse,
)
from backend.services.quiz_gen import generate_quiz, score_submission
from backend.services.weak_topics import WEAK_THRESHOLD, weak_topics_report
from backend.utils import db

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["quiz"])


@router.post("/quiz", response_model=QuizGenerateResponse)
def create_quiz(payload: QuizGenerateRequest) -> QuizGenerateResponse:
    """Generate an MCQ quiz from an uploaded PDF.

    Args:
        payload: Document, question count, difficulty and optional topic.

    Returns:
        The generated ``quiz_id`` and its questions (correct answers included
        for the client to score locally, but the server re-scores on submit).
    """
    try:
        result = generate_quiz(
            doc_id=payload.doc_id,
            num_questions=payload.num_questions,
            difficulty=payload.difficulty,
            topic=payload.topic,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Quiz generation failed for %s", payload.doc_id)
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {exc}") from exc

    return QuizGenerateResponse(
        quiz_id=result["quiz_id"],
        doc_id=payload.doc_id,
        questions=[QuizQuestion(**question) for question in result["questions"]],
        topic=result["topic"],
        difficulty=result["difficulty"],
        provider=result["provider"],
        warning=result["warning"],
    )


@router.post("/quiz/submit", response_model=QuizSubmitResponse)
def submit_quiz(payload: QuizSubmitRequest) -> QuizSubmitResponse:
    """Score a quiz submission and update the weak-topic profile."""
    if not db.get_document(payload.doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document: {payload.doc_id}")
    try:
        result = score_submission(payload.doc_id, payload.quiz_id, payload.answers)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Scoring failed for quiz %s", payload.quiz_id)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    return QuizSubmitResponse(
        quiz_id=result["quiz_id"],
        score=result["score"],
        total=result["total"],
        percentage=result["percentage"],
        per_question_feedback=[
            QuestionFeedback(
                question=item["question"],
                selected=item["selected"],
                correct_index=item["correct_index"],
                is_correct=item["is_correct"],
                explanation=item["explanation"],
                source_page=item["source_page"],
                topic=item["topic"],
            )
            for item in result["per_question_feedback"]
        ],
        weak_topics=result["weak_topics"],
    )


@router.get("/weak-topics/{doc_id}", response_model=WeakTopicsResponse)
def weak_topics(doc_id: str) -> WeakTopicsResponse:
    """Return per-topic accuracy across every quiz attempt for a document."""
    if not db.get_document(doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    report = weak_topics_report(doc_id)
    return WeakTopicsResponse(
        doc_id=doc_id,
        attempts=report["attempts"],
        average_score=report["average_score"],
        topics=[
            TopicAccuracy(
                topic=row["topic"], correct=row["correct"], total=row["total"],
                accuracy=row["accuracy"],
            )
            for row in report["topics"]
        ],
        weak_topics=report["weak_topics"],
    )


@router.get("/quiz/{quiz_id}")
def get_quiz(quiz_id: str) -> dict[str, object]:
    """Return a stored quiz (used by the UI after a page refresh)."""
    quiz = db.get_quiz(quiz_id)
    if not quiz:
        raise HTTPException(status_code=404, detail=f"Unknown quiz: {quiz_id}")
    quiz.pop("questions_json", None)
    return quiz


@router.get("/quiz-stats/{doc_id}")
def quiz_stats(doc_id: str) -> dict[str, object]:
    """Compact history view: attempts, best score and the weak threshold."""
    results = db.list_quiz_results(doc_id)
    scores = [result["score"] / result["total"] for result in results if result["total"]]
    return {
        "doc_id": doc_id,
        "attempts": len(results),
        "best_percentage": round(100 * max(scores), 2) if scores else 0.0,
        "average_percentage": round(100 * sum(scores) / len(scores), 2) if scores else 0.0,
        "weak_threshold": WEAK_THRESHOLD,
        "recent": [
            {
                "quiz_id": result["quiz_id"],
                "score": result["score"],
                "total": result["total"],
                "taken_at": result["taken_at"],
            }
            for result in results[:10]
        ],
    }