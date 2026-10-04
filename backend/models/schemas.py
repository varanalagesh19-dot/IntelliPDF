"""Pydantic v2 request/response models for every IntelliPDF endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from backend.services.mark_wise import MARK_TYPES, normalise_mark_type
from backend.utils.language import SUPPORTED_LANGUAGES, normalize_language

Difficulty = Literal["easy", "medium", "hard"]
RevisionMode = Literal["last_minute", "definitions", "key_concepts"]


# ── Shared ───────────────────────────────────────────────────────────────


class SourceRef(BaseModel):
    """A verifiable citation: which page a piece of the answer came from."""

    page: int = Field(..., ge=0, description="1-based PDF page number")
    chunk_id: str = ""
    heading: str = ""
    snippet: str = ""
    score: float = 0.0


class DocumentInfo(BaseModel):
    """Metadata about one uploaded PDF."""

    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    uploaded_at: str = ""


# ── Upload ───────────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Result of ``POST /upload``."""

    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    status: str = "processed"
    syllabus_chars: int = 0
    message: str = ""


# ── Ask ──────────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    doc_id: str
    question: str = Field(..., min_length=1, max_length=1000)
    mark_type: str = "5M"
    language: str = "english"

    @field_validator("mark_type")
    @classmethod
    def _check_mark(cls, value: str) -> str:
        return normalise_mark_type(value)

    @field_validator("language")
    @classmethod
    def _check_language(cls, value: str) -> str:
        return normalize_language(value)


class AskResponse(BaseModel):
    """Mark-wise, source-grounded answer."""

    answer: str
    sources: list[SourceRef] = []
    confidence: float = 0.0
    confidence_label: str = "low"
    provider: str = ""
    model: str = ""
    mark_type: str = "5M"
    language: str = "english"
    grounded: bool = True
    elapsed_ms: int = 0


# ── Quiz ─────────────────────────────────────────────────────────────────


class QuizQuestion(BaseModel):
    """A single generated multiple-choice question."""

    question: str
    options: list[str]
    correct_index: int
    explanation: str = ""
    source_page: int = 0
    topic: str = ""
    difficulty: str = "medium"


class QuizGenerateRequest(BaseModel):
    """Request body for ``POST /quiz``."""

    doc_id: str
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: Difficulty = "medium"
    topic: str = ""


class QuizGenerateResponse(BaseModel):
    """Generated quiz with its stored ``quiz_id``."""

    quiz_id: str
    doc_id: str
    questions: list[QuizQuestion]
    topic: str = ""
    difficulty: str = "medium"
    provider: str = ""
    warning: str = ""


class QuizSubmitRequest(BaseModel):
    """Request body for ``POST /quiz/submit``."""

    doc_id: str
    quiz_id: str
    answers: list[int] = Field(default_factory=list, description="Selected option index per question")


class QuestionFeedback(BaseModel):
    """Per-question result shown after submission."""

    question: str
    selected: int
    correct_index: int
    is_correct: bool
    explanation: str = ""
    source_page: int = 0
    topic: str = ""


class QuizSubmitResponse(BaseModel):
    """Score, per-question feedback and the derived weak topics."""

    quiz_id: str
    score: int
    total: int
    percentage: float
    per_question_feedback: list[QuestionFeedback] = []
    weak_topics: list[str] = []


# ── Evaluate ─────────────────────────────────────────────────────────────


class EvaluateRequest(BaseModel):
    """Request body for ``POST /evaluate``."""

    doc_id: str
    question: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    mark_type: str = "5M"

    @field_validator("mark_type")
    @classmethod
    def _check_mark(cls, value: str) -> str:
        return normalise_mark_type(value)


class EvaluateResponse(BaseModel):
    """Model answer compared against the student's attempt."""

    score_out_of: int
    score_earned: float
    percentage: float
    missing_points: list[str] = []
    incorrect_points: list[str] = []
    suggestions: list[str] = []
    model_answer: str = ""
    sources: list[SourceRef] = []


# ── Study plan ───────────────────────────────────────────────────────────


class StudyPlanRequest(BaseModel):
    """Request body for ``POST /study-plan``."""

    doc_id: str
    exam_date: str = Field(..., description="ISO date, e.g. 2026-11-20")
    daily_hours: float = Field(3.0, gt=0, le=14)
    weak_topics: list[str] = Field(default_factory=list)
    language: str = "english"

    @field_validator("language")
    @classmethod
    def _check_language(cls, value: str) -> str:
        return normalize_language(value)


class StudyPlanDay(BaseModel):
    """One day of the generated plan."""

    day_num: int
    date: str
    topics: list[dict[str, Any]] = []
    total_hours: float = 0.0
    focus: str = ""


class StudyPlanResponse(BaseModel):
    """Day-wise study plan."""

    doc_id: str
    exam_date: str
    days_left: int
    daily_hours: float
    total_hours: float
    topics: list[str] = []
    plan: list[StudyPlanDay] = []
    disclaimer: str = ""


# ── Mock exam ────────────────────────────────────────────────────────────


class MockExamRequest(BaseModel):
    """Request body for ``POST /mock-exam``."""

    doc_id: str
    duration_minutes: int = Field(60, ge=5, le=240)
    marks_distribution: dict[str, int] = Field(default_factory=dict)
    language: str = "english"


class MockExamQuestion(BaseModel):
    """One question in the mock paper (mark-wise, descriptive)."""

    question: str
    mark_type: str
    marks: int
    topic: str = ""
    model_answer: str = ""
    source_page: int = 0


class MockExamResponse(BaseModel):
    """Timed mock paper plus the answer key."""

    exam_id: str
    doc_id: str
    duration_minutes: int
    total_marks: int
    starts_at: str
    questions: list[MockExamQuestion] = []
    answer_key: dict[str, str] = {}
    instructions: str = ""


# ── Revision ─────────────────────────────────────────────────────────────


class RevisionRequest(BaseModel):
    """Request body for ``POST /revision``."""

    doc_id: str
    mode: RevisionMode = "last_minute"
    num_cards: int = Field(10, ge=1, le=30)
    language: str = "english"

    @field_validator("language")
    @classmethod
    def _check_language(cls, value: str) -> str:
        return normalize_language(value)


class Flashcard(BaseModel):
    """A single revision card."""

    front: str
    back: str
    source_page: int = 0


class RevisionResponse(BaseModel):
    """Flashcards for the requested revision mode."""

    doc_id: str
    mode: str
    cards: list[Flashcard] = []
    sources: list[SourceRef] = []


# ── Weak topics ──────────────────────────────────────────────────────────


class TopicAccuracy(BaseModel):
    """Accuracy for one topic across all attempts."""

    topic: str
    correct: int
    total: int
    accuracy: float


class WeakTopicsResponse(BaseModel):
    """Aggregated quiz performance per topic."""

    doc_id: str
    attempts: int
    average_score: float
    topics: list[TopicAccuracy] = []
    weak_topics: list[str] = []


# ── Health ───────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Result of ``GET /health``."""

    status: str = "ok"
    services: dict[str, Any] = {}
    version: str = "1.0.0"


__all__ = [
    "AskRequest",
    "AskResponse",
    "DocumentInfo",
    "Difficulty",
    "EvaluateRequest",
    "EvaluateResponse",
    "Flashcard",
    "HealthResponse",
    "MARK_TYPES",
    "MockExamQuestion",
    "MockExamRequest",
    "MockExamResponse",
    "QuestionFeedback",
    "QuizGenerateRequest",
    "QuizGenerateResponse",
    "QuizQuestion",
    "QuizSubmitRequest",
    "QuizSubmitResponse",
    "RevisionMode",
    "RevisionRequest",
    "RevisionResponse",
    "SourceRef",
    "StudyPlanDay",
    "StudyPlanRequest",
    "StudyPlanResponse",
    "SUPPORTED_LANGUAGES",
    "TopicAccuracy",
    "UploadResponse",
    "WeakTopicsResponse",
]