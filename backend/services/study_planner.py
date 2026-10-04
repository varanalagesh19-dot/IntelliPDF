"""AI study planner.

Turns "exam on the 24th, I can study 3 hours a day" into a day-wise schedule:

1. read the document structure (headings) + optional syllabus as the topic list,
2. merge in the student's weak topics from quiz history,
3. weight topics - weak x1.5, syllabus/important x1.2, normal x1.0,
4. distribute the weighted topics into ``days_left * daily_hours`` of study time,
   splitting any topic that is bigger than one day across consecutive days,
5. reserve the final day for revision and a mock-test recap.

The output is fully deterministic, so a plan can be regenerated and compared.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from backend.services.vector_store import get_vector_store
from backend.services.weak_topics import get_weak_topics
from backend.utils import db
from backend.utils.metadata import iso_now

LOGGER = logging.getLogger(__name__)

WEIGHT_WEAK = 1.5
WEIGHT_IMPORTANT = 1.2
WEIGHT_NORMAL = 1.0

#: Fallback topic list when a document exposes no headings at all.
FALLBACK_TOPICS = [
    "Full document read-through",
    "Definitions and key terms",
    "Important formulas and diagrams",
    "Previous year questions",
]

_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[\.\)])\s+(.{3,80})$")
#: Matches characters that are *not* part of a readable topic name.
_WORD_RE = re.compile(r"[^A-Za-z0-9 ]+")


def _clean_topic(text: str) -> str:
    """Keep only word characters and normalise whitespace."""
    cleaned = _WORD_RE.sub(" ", (text or "").replace("—", " ").replace(":", " "))
    cleaned = " ".join(cleaned.split()).strip(" -–—")
    return cleaned if 3 <= len(cleaned) <= 70 else ""


def parse_syllabus(syllabus: str, limit: int = 25) -> list[str]:
    """Extract important topics from free-text syllabus lines."""
    topics: list[str] = []
    for line in (syllabus or "").splitlines():
        match = _BULLET_RE.match(line) or (
            _clean_topic(line) and line.strip()
        )
        candidate = _clean_topic(match if isinstance(match, str) else match.group(1))
        if candidate and candidate.lower() not in {t.lower() for t in topics}:
            topics.append(candidate)
        if len(topics) >= limit:
            break
    return topics


def document_headings(doc_id: str, limit: int = 30) -> list[str]:
    """Return the distinct section headings stored for a document."""
    try:
        meta = get_vector_store().load(doc_id)["meta"]
    except Exception as exc:  # noqa: BLE001 - fall back to an empty list
        LOGGER.warning("Could not read headings for %s: %s", doc_id, exc)
        return []
    headings: list[str] = []
    for chunk in meta:
        heading = _clean_topic(chunk.get("heading") or "")
        if heading and heading.lower() not in {h.lower() for h in headings}:
            headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def build_topic_pool(
    doc_id: str,
    syllabus: str = "",
    weak_topics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the weighted topic pool for planning.

    Args:
        doc_id: Document identifier.
        syllabus: Optional syllabus text typed at upload time.
        weak_topics: Topics the student is weak at (from quiz history).

    Returns:
        ``[{"name", "weight", "priority", "reason"}]`` sorted by weight.
    """
    weak = {_clean_topic(topic).lower() for topic in (weak_topics or [])}
    weak.discard("")

    syllabus_topics = parse_syllabus(syllabus)
    headings = document_headings(doc_id)

    pool: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(name: str, weight: float, priority: str, reason: str) -> None:
        cleaned = _clean_topic(name)
        key = cleaned.lower()
        if not cleaned or key in seen:
            return
        seen.add(key)
        pool.append(
            {"name": cleaned, "weight": weight, "priority": priority, "reason": reason}
        )

    for topic in weak_topics or []:
        add(topic, WEIGHT_WEAK, "high", "weak in recent quizzes")
    for topic in syllabus_topics:
        if topic.lower() in weak:
            continue
        add(topic, WEIGHT_IMPORTANT, "medium", "from your syllabus")
    for heading in headings:
        if heading.lower() in weak:
            continue
        add(heading, WEIGHT_NORMAL, "low", "section in the PDF")

    if not pool:
        for topic in FALLBACK_TOPICS:
            add(topic, WEIGHT_NORMAL, "low", "default plan")

    pool.sort(key=lambda item: (-item["weight"], item["name"]))
    return pool


def _revision_day(exam_date: date, weak_topics: list[str]) -> dict[str, Any]:
    weak = weak_topics[:4] or ["weak areas"]
    return {
        "day_num": 0,  # filled in by the caller
        "date": "",
        "focus": "🔥 Final revision",
        "total_hours": 0.0,
        "topics": [
            {
                "name": "Re-read all definitions & formulas",
                "hours": 1.0,
                "priority": "high",
                "reason": "last-minute recall",
            },
            {
                "name": "Revise weak topics: " + ", ".join(weak),
                "hours": 1.5,
                "priority": "high",
                "reason": "weak in recent quizzes",
            },
            {
                "name": "Solve one previous-year paper",
                "hours": 1.0,
                "priority": "medium",
                "reason": "exam simulation",
            },
        ],
    }


def generate_study_plan(
    doc_id: str,
    exam_date: str,
    daily_hours: float = 3.0,
    weak_topics: list[str] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Build a day-wise study plan ending on ``exam_date``.

    Args:
        doc_id: Document identifier.
        exam_date: ISO date string of the exam.
        daily_hours: Study hours available per day.
        weak_topics: Optional override; defaults to stored weak topics.
        today: Injectable "current date" (used by tests).

    Returns:
        ``{"doc_id", "exam_date", "days_left", "daily_hours", "total_hours",
        "topics", "plan"}`` where ``plan`` is one entry per day.

    Raises:
        ValueError: The exam date is invalid or already in the past.
    """
    try:
        exam = datetime.strptime(str(exam_date).strip()[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"exam_date must look like 2026-11-20 (got {exam_date!r}).") from exc

    start = today or date.today()
    days_left = (exam - start).days
    if days_left < 0:
        raise ValueError(f"The exam date {exam} is in the past.")
    if days_left == 0:
        raise ValueError("The exam is today - use Revision or Exam Mode instead.")

    document = db.get_document(doc_id) or {}
    syllabus = document.get("syllabus") or ""
    weak = weak_topics if weak_topics else get_weak_topics(doc_id)
    pool = build_topic_pool(doc_id, syllabus, weak)

    # The last day is reserved for revision whenever there is more than one day.
    content_days = max(1, days_left - 1) if days_left > 1 else 1
    budget = round(content_days * max(0.5, daily_hours), 2)

    weights = sum(item["weight"] for item in pool) or 1.0
    allotments = [
        {
            **item,
            "hours": round(max(0.25, budget * item["weight"] / weights), 2),
        }
        for item in pool
    ]

    plan: list[dict[str, Any]] = []
    pending = [dict(topic) for topic in allotments]
    cursor = 0
    for offset in range(content_days):
        day_date = start + timedelta(days=offset)
        capacity = max(0.5, daily_hours)
        day_topics: list[dict[str, Any]] = []
        used = 0.0
        part = 1
        # Fill each day from the weighted topic list; a topic larger than the
        # daily capacity is continued on the following day.
        while cursor < len(pending) and used < capacity - 0.05:
            topic = pending[cursor]
            remaining = round(topic["hours"], 2)
            slice_hours = round(min(remaining, capacity - used), 2)
            if slice_hours <= 0.05:
                cursor += 1
                part = 1
                continue
            day_topics.append(
                {
                    "name": topic["name"],
                    "hours": slice_hours,
                    "priority": topic["priority"],
                    "reason": topic["reason"],
                    "part": part,
                }
            )
            used = round(used + slice_hours, 2)
            topic["hours"] = round(remaining - slice_hours, 2)
            if topic["hours"] <= 0.05:
                cursor += 1
                part = 1
            else:
                part += 1
        plan.append(
            {
                "day_num": offset + 1,
                "date": day_date.isoformat(),
                "focus": (
                    f"Day {offset + 1} - "
                    + (", ".join(t["name"] for t in day_topics[:2]) or "Self-test & recall")
                ),
                "topics": day_topics,
                "total_hours": round(sum(t["hours"] for t in day_topics), 2),
            }
        )

    # The last day before the exam is always reserved for revision, and any
    # leftover topic hours are folded into it.
    if days_left > 1:
        revision = _revision_day(exam, weak)
        for topic in [t for t in pending[cursor:] if t["hours"] > 0.05][:3]:
            revision["topics"].append(
                {
                    "name": topic["name"],
                    "hours": round(min(topic["hours"], daily_hours / 2), 2),
                    "priority": topic["priority"],
                    "reason": topic["reason"],
                }
            )
        revision["day_num"] = days_left
        revision["date"] = exam.isoformat()
        revision["total_hours"] = round(sum(t["hours"] for t in revision["topics"]), 2)
        plan.append(revision)

    plan.sort(key=lambda day: day["day_num"])

    result = {
        "doc_id": doc_id,
        "exam_date": exam.isoformat(),
        "days_left": days_left,
        "daily_hours": round(daily_hours, 2),
        "total_hours": round(sum(day["total_hours"] for day in plan), 2),
        "topics": [item["name"] for item in allotments],
        "weak_topics": weak,
        "weights": {"weak": WEIGHT_WEAK, "syllabus": WEIGHT_IMPORTANT, "normal": WEIGHT_NORMAL},
        "plan": plan,
        "created_at": iso_now(),
        "disclaimer": (
            "Identifies topics for focused study; does not predict the actual "
            "question paper."
        ),
    }

    plan_id = f"plan_{uuid.uuid4().hex[:10]}"
    db.save_study_plan(plan_id, doc_id, result, exam.isoformat())
    LOGGER.info(
        "Generated plan %s for %s: %d days, %.1f h total",
        plan_id,
        doc_id,
        len(plan),
        result["total_hours"],
    )
    result["plan_id"] = plan_id
    return result