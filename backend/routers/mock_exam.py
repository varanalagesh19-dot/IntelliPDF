"""``POST /mock-exam`` - timed simulation of a real question paper.

The endpoint builds a paper with a configurable marks distribution
(``{"1M": 5, "2M": 3, "5M": 2, "16M": 1}``), stores the session in memory with
its start timestamp, and returns the questions plus a separate answer key.

Submission is graded per question through the same evaluator used by
``/evaluate``; if the wall-clock time exceeds the allotted duration the paper
is auto-submitted and flagged.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.models.schemas import MockExamQuestion, MockExamRequest, MockExamResponse
from backend.services.chunker import iter_text_fragments
from backend.services.evaluator import evaluate_answer
from backend.services.llm_client import get_llm_client, parse_json_block
from backend.services.mark_wise import mark_int, marks_distribution, normalise_mark_type
from backend.services.rag_pipeline import SYSTEM_ROLE, extractive_answer
from backend.services.revision import sample_chunks
from backend.utils import db
from backend.utils.metadata import iso_now

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["exam"])

SYSTEM_ROLE_EXAM = (
    "You are a university examiner who sets a model question paper. You write "
    "only from the supplied material and you always reply with valid JSON."
)

DEFAULT_DISTRIBUTION: dict[str, int] = {"1M": 5, "2M": 3, "5M": 2, "16M": 1}

#: exam_id -> session state (single-process demo; swap for Redis in production).
_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_LOCK = threading.Lock()
MAX_SESSION_AGE_SECONDS = 6 * 60 * 60


class MockExamSubmitRequest(BaseModel):
    """Body for ``POST /mock-exam/submit``."""

    exam_id: str
    answers: dict[str, str] = Field(
        default_factory=dict, description="question_id -> student answer"
    )


class MockExamQuestionResult(BaseModel):
    """Graded result for a single mock-exam question."""

    question_id: str
    question: str
    mark_type: str
    score_earned: float
    score_out_of: int
    missing_points: list[str] = []
    suggestions: list[str] = []
    source_page: int = 0


class MockExamSubmitResponse(BaseModel):
    """Overall mock-exam result."""

    exam_id: str
    score: float
    total_marks: int
    percentage: float
    time_used_minutes: float
    auto_submitted: bool
    results: list[MockExamQuestionResult] = []


def build_exam_prompt(context: str, mark_type: str, count: int, page_hint: int) -> str:
    """Prompt for one mark-type block of the question paper."""
    marks = mark_int(mark_type)
    return (
        f"MATERIAL (around page {page_hint}):\n{context}\n\n"
        f"Write {count} exam question(s), each worth exactly {marks} mark"
        f"{'s' if marks != 1 else ''}, answerable ONLY from the material "
        f"above. Aim for this shape: {marks_distribution(mark_type)}.\n"
        "Also write the model answer for each question.\n\n"
        'Reply with ONLY this JSON (no code fence):\n'
        '[{"question": "...", "model_answer": "...", "topic": "...", '
        f'"source_page": {page_hint}}}]'
    )


def _chunk_topic(chunk: dict[str, Any]) -> str:
    """Derive a readable topic label from a chunk's heading or content."""
    heading = chunk.get("heading") or ""
    if heading:
        return heading.split("—")[0].strip()[:60]
    # Fall back to the most distinctive phrase of the chunk itself.
    best = ""
    for fragment in iter_text_fragments(chunk.get("text", "")):
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", fragment) if len(w) > 3]
        if len(words) >= 2:
            best = " ".join(words[:3])
            break
    return best or f"the material on page {chunk['page_num']}"


def _offline_questions(chunks: list[dict[str, Any]], mark_type: str, count: int
                       ) -> list[dict[str, Any]]:
    """Deterministic questions from chunk headings when no LLM is reachable."""
    questions: list[dict[str, Any]] = []
    used: set[str] = set()
    for index in range(count):
        chunk = chunks[index % len(chunks)]
        topic = _chunk_topic(chunk)
        if topic.lower() in used:
            topic = f"{topic} (part {index + 1})"
        used.add(topic.lower())
        scored = [{"chunk": chunk, "score": 0.9, "rank": 1}]
        questions.append(
            {
                "question": f"Explain {topic}.",
                "model_answer": extractive_answer(
                    f"Explain {topic}", scored, mark_type, "english"
                ),
                "topic": topic[:40],
                "source_page": int(chunk["page_num"]),
            }
        )
    return questions


def _build_block(
    chunks: list[dict[str, Any]], mark_type: str, count: int, offset: int
) -> list[dict[str, Any]]:
    """Build ``count`` questions of one mark type (LLM, else offline)."""
    slice_chunks = chunks[offset % max(1, len(chunks)) :][: max(count * 2, 3)] or chunks[:3]
    context = "\n\n---\n\n".join(
        f"[Page {chunk['page_num']}]\n{chunk['text']}" for chunk in slice_chunks
    )
    prompt = build_exam_prompt(context, mark_type, count, offset + 1)
    response = get_llm_client().generate(
        prompt, system=SYSTEM_ROLE_EXAM, max_tokens=2400
    )

    questions: list[dict[str, Any]] = []
    if response.ok:
        payload = parse_json_block(response.text)
        if isinstance(payload, dict):
            payload = payload.get("questions") or []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question") or "").strip()
                answer = str(item.get("model_answer") or item.get("answer") or "").strip()
                if not question or not answer:
                    continue
                try:
                    page = int(item.get("source_page") or slice_chunks[0]["page_num"])
                except (TypeError, ValueError, IndexError):
                    page = int(slice_chunks[0]["page_num"]) if slice_chunks else 0
                questions.append(
                    {
                        "question": question,
                        "model_answer": answer,
                        "topic": str(item.get("topic") or "")[:40],
                        "source_page": page,
                    }
                )
    if len(questions) < count:
        questions.extend(_offline_questions(slice_chunks, mark_type, count - len(questions)))
    return questions[:count]


@router.post("/mock-exam", response_model=MockExamResponse)
def create_mock_exam(payload: MockExamRequest) -> MockExamResponse:
    """Create a timed mock exam with a configurable marks distribution."""
    if not db.get_document(payload.doc_id):
        raise HTTPException(status_code=404, detail=f"Unknown document: {payload.doc_id}")
    try:
        chunks = sample_chunks(payload.doc_id, 12)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not chunks:
        raise HTTPException(status_code=400, detail="This document has no indexed content.")

    distribution = {
        normalise_mark_type(mark): max(0, int(count))
        for mark, count in (payload.marks_distribution or DEFAULT_DISTRIBUTION).items()
    }
    distribution = {mark: count for mark, count in distribution.items() if count > 0}
    if not distribution:
        distribution = {"5M": 5}

    questions: list[MockExamQuestion] = []
    offset = 0
    for mark_type, count in distribution.items():
        for item in _build_block(chunks, mark_type, count, offset):
            offset += 1
            questions.append(
                MockExamQuestion(
                    question=item["question"],
                    mark_type=mark_type,
                    marks=mark_int(mark_type),
                    topic=item.get("topic", ""),
                    model_answer=item.get("model_answer", ""),
                    source_page=int(item.get("source_page", 0)),
                )
            )

    exam_id = f"exam_{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(timezone.utc)
    total_marks = sum(question.marks for question in questions)
    answer_key = {
        f"q{index + 1}": question.model_answer
        for index, question in enumerate(questions)
    }

    with _SESSION_LOCK:
        _prune_sessions()
        _SESSIONS[exam_id] = {
            "doc_id": payload.doc_id,
            "started_at": started_at,
            "duration_minutes": payload.duration_minutes,
            "total_marks": total_marks,
            "questions": [question.model_dump() for question in questions],
            "submitted": False,
        }

    LOGGER.info(
        "Mock exam %s created: %d questions, %d marks, %d minutes",
        exam_id,
        len(questions),
        total_marks,
        payload.duration_minutes,
    )
    return MockExamResponse(
        exam_id=exam_id,
        doc_id=payload.doc_id,
        duration_minutes=payload.duration_minutes,
        total_marks=total_marks,
        starts_at=started_at.isoformat(timespec="seconds"),
        questions=questions,
        answer_key=answer_key,
        instructions=(
            f"You have {payload.duration_minutes} minutes for {total_marks} marks. "
            "Answer every question in the space you would get in the hall; the "
            "timer is enforced server-side when you submit."
        ),
    )


@router.get("/mock-exam/{exam_id}")
def exam_state(exam_id: str) -> dict[str, Any]:
    """Return the live state of a mock exam (time left, submitted or not)."""
    session = _SESSIONS.get(exam_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown exam: {exam_id}")
    elapsed = (datetime.now(timezone.utc) - session["started_at"]).total_seconds()
    remaining = max(0.0, session["duration_minutes"] * 60 - elapsed)
    return {
        "exam_id": exam_id,
        "doc_id": session["doc_id"],
        "time_remaining_seconds": round(remaining, 1),
        "elapsed_seconds": round(elapsed, 1),
        "submitted": session["submitted"],
        "total_marks": session["total_marks"],
        "started_at": session["started_at"].isoformat(timespec="seconds"),
        "server_time": iso_now(),
    }


@router.post("/mock-exam/submit", response_model=MockExamSubmitResponse)
def submit_mock_exam(payload: MockExamSubmitRequest) -> MockExamSubmitResponse:
    """Grade a mock paper; auto-submit and flag if the timer already expired."""
    session = _SESSIONS.get(payload.exam_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown exam: {payload.exam_id}")
    if session["submitted"]:
        raise HTTPException(status_code=409, detail="This exam was already submitted.")

    elapsed_minutes = (
        datetime.now(timezone.utc) - session["started_at"]
    ).total_seconds() / 60
    auto_submitted = elapsed_minutes > session["duration_minutes"]

    results: list[MockExamQuestionResult] = []
    earned_total = 0.0
    for index, question in enumerate(session["questions"], start=1):
        question_id = f"q{index}"
        answer = (payload.answers.get(question_id) or "").strip()
        marks = int(question["marks"])
        if not answer:
            results.append(
                MockExamQuestionResult(
                    question_id=question_id,
                    question=question["question"],
                    mark_type=question["mark_type"],
                    score_earned=0.0,
                    score_out_of=marks,
                    missing_points=["Not attempted"],
                    suggestions=["Attempt every question - a rough answer earns more "
                                 "than a blank page."],
                    source_page=int(question.get("source_page", 0)),
                )
            )
            continue
        try:
            graded = evaluate_answer(
                doc_id=session["doc_id"],
                question=question["question"],
                student_answer=answer,
                mark_type=question["mark_type"],
            )
        except Exception as exc:  # noqa: BLE001 - never lose the whole paper
            LOGGER.warning("Grading %s failed: %s", question_id, exc)
            graded = {
                "score_earned": 0.0,
                "missing_points": ["Grading unavailable for this question"],
                "suggestions": [],
            }
        earned_total += float(graded["score_earned"])
        results.append(
            MockExamQuestionResult(
                question_id=question_id,
                question=question["question"],
                mark_type=question["mark_type"],
                score_earned=round(float(graded["score_earned"]), 1),
                score_out_of=marks,
                missing_points=list(graded.get("missing_points") or []),
                suggestions=list(graded.get("suggestions") or []),
                source_page=int(question.get("source_page", 0)),
            )
        )

    session["submitted"] = True
    total_marks = int(session["total_marks"])
    LOGGER.info(
        "Mock exam %s submitted: %.1f/%d in %.1f min%s",
        payload.exam_id,
        earned_total,
        total_marks,
        elapsed_minutes,
        " (auto-submitted)" if auto_submitted else "",
    )
    return MockExamSubmitResponse(
        exam_id=payload.exam_id,
        score=round(earned_total, 1),
        total_marks=total_marks,
        percentage=round(100.0 * earned_total / max(1, total_marks), 2),
        time_used_minutes=round(elapsed_minutes, 1),
        auto_submitted=auto_submitted,
        results=results,
    )


def _prune_sessions() -> None:
    """Drop exam sessions older than :data:`MAX_SESSION_AGE_SECONDS`."""
    now = datetime.now(timezone.utc)
    stale = [
        exam_id
        for exam_id, session in _SESSIONS.items()
        if (now - session["started_at"]).total_seconds() > MAX_SESSION_AGE_SECONDS
    ]
    for exam_id in stale:
        _SESSIONS.pop(exam_id, None)


@router.get("/mock-exam/{exam_id}/answer-key")
def answer_key(exam_id: str) -> dict[str, str]:
    """Return the model answers for a submitted exam (post-exam revision)."""
    session = _SESSIONS.get(exam_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Unknown exam: {exam_id}")
    if not session["submitted"]:
        raise HTTPException(
            status_code=409, detail="Submit the paper before opening the answer key."
        )
    return {
        f"q{index + 1}": question["model_answer"]
        for index, question in enumerate(session["questions"])
    }