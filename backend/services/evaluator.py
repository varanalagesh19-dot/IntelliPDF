"""Answer evaluation: compare a student's answer to the source material.

Two paths, one contract:

* **LLM path** - retrieve the chunks that answer the question, ask the model to
  compare the student's answer with a model answer and reply with strict JSON
  ``{score_out_of, missing_points, incorrect_points, suggestions}``.
* **Offline path** - deterministic lexical scoring: coverage of the key terms in
  the source (reward), contradictions (penalty) and a list of source key points
  the student failed to mention. Fully explainable, no API key needed.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from backend.services.llm_client import as_list_of_text, get_llm_client, parse_json_block
from backend.services.mark_wise import mark_int, normalise_mark_type
from backend.services.rag_pipeline import (
    SYSTEM_ROLE,
    build_context,
    extractive_answer,
    retrieve,
)
from backend.utils.metadata import iso_now, truncate

LOGGER = logging.getLogger(__name__)

SYSTEM_ROLE_EVALUATOR = (
    "You are a strict but kind exam examiner. You compare a student's answer "
    "with the reference material, you never invent content, and you always "
    "reply with valid JSON."
)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "for",
    "to", "and", "or", "with", "by", "as", "at", "it", "its", "this", "that",
    "be", "been", "can", "will", "which", "what", "we", "you", "they", "there",
}


def build_evaluation_prompt(
    question: str,
    student_answer: str,
    marks: int,
    context: str,
    model_answer: str,
) -> str:
    """Compose the examiner prompt for one submission."""
    return (
        f"REFERENCE CONTEXT (from the student's PDF):\n{context}\n\n"
        f"MODEL ANSWER (what a full-mark answer contains):\n{model_answer}\n\n"
        f"QUESTION ({marks} marks): {question}\n\n"
        f"STUDENT ANSWER:\n{student_answer}\n\n"
        "Evaluate strictly and fairly:\n"
        "1. Award marks for each point the student covered correctly.\n"
        "2. List the points from the model answer that the student MISSED.\n"
        "3. List anything the student wrote that is factually WRONG per the "
        "context.\n"
        "4. Give 2-4 concrete suggestions for improving the next attempt.\n"
        "5. Be strict: vague or copied-but-unrelated text scores low.\n\n"
        'Reply with ONLY this JSON (no code fence):\n'
        '{"score_out_of": 5, "score_earned": 3.5, "missing_points": ["..."], '
        '"incorrect_points": ["..."], "suggestions": ["..."]}'
    )


# ── Deterministic scoring ────────────────────────────────────────────────


def _content_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (text or "").lower())
        if word not in STOPWORDS
    ]


def _sentences(text: str) -> list[str]:
    """Split text into trimmed sentences (table fragments handled upstream)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _key_phrases(text: str, limit: int = 12) -> list[str]:
    """Extract candidate key phrases (noun-ish word pairs) from source text."""
    phrases: list[str] = []
    for sentence in _sentences(text)[:20]:
        words = _content_words(sentence)
        for size in (3, 2):
            for start in range(0, max(0, len(words) - size + 1), size):
                phrase = " ".join(words[start : start + size])
                if len(phrase) > 8 and phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= limit * 2:
                    return phrases[: limit * 2]
    return phrases[: limit * 2]


def _reference_points(context: str, question: str, limit: int = 5) -> list[str]:
    """Return the source sentences a full answer would have to cover.

    Grading against whole sentences (instead of arbitrary word windows) makes
    "missing points" something a student can actually act on: each one is a
    real statement from their PDF, ordered by how strongly it answers the
    question.
    """
    query_terms = set(_content_words(question))
    scored: list[tuple[float, str]] = []
    for sentence in _sentences(context):
        terms = set(_content_words(sentence))
        if not terms or len(sentence.split()) < 5:
            continue
        overlap = len(terms & query_terms)
        if query_terms and not overlap:
            continue
        scored.append((overlap / max(1, len(query_terms)) + len(terms) / 200.0, sentence))
    if not scored:
        scored = [(0.0, sentence) for sentence in _sentences(context)[:limit]]
    scored.sort(key=lambda item: -item[0])

    points: list[str] = []
    for _score, sentence in scored:
        if len(sentence) > 220:
            sentence = sentence[:217].rsplit(" ", 1)[0] + "..."
        if sentence not in points:
            points.append(sentence)
        if len(points) >= limit:
            break
    return points


def _is_covered(point: str, student_words: set[str]) -> bool:
    """Return ``True`` when the student's answer reproduces this point."""
    point_words = set(_content_words(point))
    if not point_words:
        return False
    shared = len(point_words & student_words) / len(point_words)
    return shared >= 0.6


def offline_evaluate(
    question: str,
    student_answer: str,
    marks: int,
    context: str,
    model_answer: str,
) -> dict[str, Any]:
    """Score an answer with lexical overlap against the reference material."""
    student_words = set(_content_words(student_answer))
    reference = f"{context}\n{model_answer}"

    # Reference points are whole source sentences the answer should cover.
    points = _reference_points(reference, question, limit=5)
    covered = [point for point in points if _is_covered(point, student_words)]
    missing = [point for point in points if point not in covered]

    coverage = len(covered) / max(1, len(points)) if points else 0.0

    # Rewards: correct terminology echoed from the source.
    reference_words = set(_content_words(reference))
    precision = len(student_words & reference_words) / max(1, len(student_words))

    # A 5-mark answer needs substance; a single line cannot earn full marks
    # just because it reuses one term from the source.
    length_words = len(student_answer.split())
    expected_words = max(40, marks * 22)
    adequacy = min(1.0, len(student_words) / max(4.0, 0.4 * expected_words))

    relevance = (
        0.65 * min(1.0, coverage / 0.5) + 0.20 * precision + 0.15 * adequacy
    )

    # Penalty: contradictions and rambling.
    contradictions = re.findall(
        r"[^.!?]*\b(not|never|cannot|isn't|doesn't|don't)\b[^.!?]*[.!?]",
        student_answer or "",
        flags=re.IGNORECASE,
    )
    length_penalty = 0.15 if length_words > expected_words * 2.2 else 0.0
    penalty = min(0.45, 0.25 * len(contradictions) + length_penalty)

    score_ratio = max(0.0, min(1.0, relevance - penalty))
    earned = round(marks * score_ratio, 1)

    suggestions: list[str] = []
    if missing:
        suggestions.append(
            "Your answer missed "
            f"{len(missing)} point(s) from the source: " + "; ".join(missing[:2])
        )
    if not student_words & reference_words:
        suggestions.append(
            "Your answer used none of the terminology from the material - use the "
            "exact terms used in the PDF."
        )
    if length_words < expected_words * 0.35:
        suggestions.append(
            f"For a {marks}-mark answer, write roughly {expected_words} words; "
            f"you wrote {length_words}."
        )
    if contradictions:
        suggestions.append(
            "Re-check statements joined by 'not/never' - negated facts are often "
            "wrong by accident."
        )
    if not suggestions:
        suggestions.append(
            "Good coverage. Practise writing the same points as crisp bullets "
            "under headings to save time in the exam."
        )

    return {
        "score_out_of": marks,
        "score_earned": earned,
        "missing_points": missing,
        "incorrect_points": [truncate(item, 160) for item in contradictions[:3]],
        "suggestions": suggestions[:4],
        "method": "offline-lexical",
    }


# ── Public API ───────────────────────────────────────────────────────────


def evaluate_answer(
    doc_id: str,
    question: str,
    student_answer: str,
    mark_type: str = "5M",
    k: int = 5,
) -> dict[str, Any]:
    """Grade a student's answer against the uploaded source.

    Args:
        doc_id: Document identifier.
        question: The exam question.
        student_answer: What the student wrote.
        mark_type: ``1M`` ... ``16M``.
        k: Chunks retrieved as the reference context.

    Returns:
        ``{"score_out_of", "score_earned", "percentage", "missing_points",
        "incorrect_points", "suggestions", "model_answer", "sources"}``.

    Raises:
        ValueError: The student answer is empty.
        FileNotFoundError: The document has no vector index.
    """
    question = (question or "").strip()
    student_answer = (student_answer or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if not student_answer:
        raise ValueError("Student answer cannot be empty.")

    mark_type = normalise_mark_type(mark_type)
    marks = mark_int(mark_type)

    results = retrieve(doc_id, question, k=k)
    if not results:
        raise FileNotFoundError(f"No retrievable content for document {doc_id}")
    context = build_context(results)

    model_answer = extractive_answer(question, results, mark_type, "english")

    prompt = build_evaluation_prompt(
        question, student_answer, marks, context, model_answer
    )
    response = get_llm_client().generate(prompt, system=SYSTEM_ROLE_EVALUATOR)

    payload: dict[str, Any] = {}
    if response.ok:
        parsed = parse_json_block(response.text)
        if isinstance(parsed, dict):
            payload = parsed

    if payload:
        try:
            earned = float(payload.get("score_earned", payload.get("score", 0)))
        except (TypeError, ValueError):
            earned = 0.0
        earned = max(0.0, min(float(marks), earned))
        result = {
            "score_out_of": marks,
            "score_earned": round(earned, 1),
            "missing_points": as_list_of_text(
                payload.get("missing_points") or payload.get("missing"), 8
            ),
            "incorrect_points": as_list_of_text(
                payload.get("incorrect_points") or payload.get("incorrect"), 6
            ),
            "suggestions": as_list_of_text(payload.get("suggestions"), 5),
            "method": f"llm:{response.provider}",
        }
    else:
        result = offline_evaluate(question, student_answer, marks, context, model_answer)

    sources = [
        {
            "page": int(result_item["chunk"]["page_num"]),
            "chunk_id": result_item["chunk"].get("chunk_id", ""),
            "heading": result_item["chunk"].get("heading") or "",
            "snippet": truncate(result_item["chunk"]["text"], 320),
            "score": round(float(result_item["score"]), 4),
        }
        for result_item in results
    ]

    result.update(
        {
            "doc_id": doc_id,
            "question": question,
            "mark_type": mark_type,
            "model_answer": model_answer,
            "sources": sources,
            "percentage": round(
                100.0 * result["score_earned"] / max(1, marks), 1
            ),
            "evaluated_at": iso_now(),
            "evaluation_id": f"eval_{uuid.uuid4().hex[:10]}",
        }
    )
    LOGGER.info(
        "Evaluated answer for %s: %s/%s (%s)",
        doc_id,
        result["score_earned"],
        marks,
        result["method"],
    )
    return result