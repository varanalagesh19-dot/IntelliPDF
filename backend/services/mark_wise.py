"""Mark-wise answer rules.

Indian university question papers score answers by marks, and an examiner
expects a *shape* for each mark count: a 1-mark answer is a single precise
line, a 16-mark answer has six labelled sections and a conclusion.  These rules
are injected verbatim into every RAG prompt so the model hits the expected
length and structure for the selected mark type.
"""

from __future__ import annotations

import logging
from typing import Final

LOGGER = logging.getLogger(__name__)

MARK_TYPES: Final[tuple[str, ...]] = ("1M", "2M", "5M", "10M", "15M", "16M")

DEFAULT_MARK: Final[str] = "5M"

#: Marks → approximate sentence budget, used by the offline extractive mode.
MARK_SENTENCE_BUDGET: Final[dict[str, int]] = {
    "1M": 2,
    "2M": 4,
    "5M": 6,
    "10M": 12,
    "15M": 16,
    "16M": 20,
}

MARK_PROMPTS: Final[dict[str, str]] = {
    "1M": (
        "MARK RULE (1 mark): Answer in 1-2 short lines only. Give the precise "
        "definition, term or formula. No examples, no preamble, no closing "
        "statement. Exactly what a student writes inside a small box."
    ),
    "2M": (
        "MARK RULE (2 marks): Answer in 3-4 short lines. Give a crisp "
        "definition plus one small example or property. Stay tight, no "
        "fluff."
    ),
    "5M": (
        "MARK RULE (5 marks): Write one structured paragraph of roughly 120 "
        "words covering 3-4 key points. Use short inline markers such as "
        "(i)/(ii)/(iii) or bullets. No introduction, no repetition, no "
        "praise, no unrelated theory."
    ),
    "10M": (
        "MARK RULE (10 marks): Write a detailed exam answer with a one-line "
        "introduction followed by 4-5 clearly labelled sections (use '##' "
        "headings or bold labels). Include at least one concrete example, a "
        "formula if relevant, and a crisp closing line. Every section must be "
        "answerable from the provided context."
    ),
    "15M": (
        "MARK RULE (15 marks): Go deeper than a 10-mark answer. Provide an "
        "introduction, 5-6 labelled sections, a worked example, a short "
        "description of the most useful diagram/flow (state it in words as "
        "'Diagram: ...'), and a real-world application for each major idea."
    ),
    "16M": (
        "MARK RULE (16 marks): Produce the most complete answer possible: "
        "introduction, definitions, 5-6 labelled sections with sub-points, "
        "multiple examples, a described diagram/flow, advantages and "
        "disadvantages (or limitations), and a short conclusion. It must read "
        "like a model answer written for a full-length paper."
    ),
}

#: Extra guidance appended for every mark type (exam hygiene).
BASE_ANSWER_RULES: Final[str] = (
    "ANSWER RULES: Answer ONLY from the provided context - never use outside "
    "knowledge and never invent facts. If the context does not contain the "
    "answer, reply exactly: 'Insufficient information in the uploaded "
    "material.' Use markdown (headings, bullets, bold) only when the mark "
    "rule asks for it. Do not mention 'the context', 'the provided text' or "
    "'the document' in the answer itself - write it as a direct answer to the "
    "examiner's question."
)


def normalise_mark_type(mark_type: str | int | None) -> str:
    """Coerce user input such as ``"10m"`` or ``10`` into a canonical key.

    Args:
        mark_type: Raw mark value from the API payload or UI.

    Returns:
        One of ``1M``, ``2M``, ``5M``, ``10M``, ``15M``, ``16M``.
    """
    if mark_type is None:
        return DEFAULT_MARK
    text = str(mark_type).strip().upper()
    if text in MARK_PROMPTS:
        return text
    if text.endswith("MARKS"):
        text = text.replace("MARKS", "").strip()
    if text.isdigit():
        candidate = f"{text}M"
        if candidate in MARK_PROMPTS:
            return candidate
    LOGGER.warning("Unknown mark type %r, defaulting to %s", mark_type, DEFAULT_MARK)
    return DEFAULT_MARK


def mark_int(mark_type: str) -> int:
    """Return the numeric value of a mark type (``"5M" -> 5``)."""
    return int(normalise_mark_type(mark_type).rstrip("M"))


def get_mark_prompt(mark_type: str | None) -> str:
    """Return the full mark instruction block injected into the prompt."""
    return MARK_PROMPTS[normalise_mark_type(mark_type)]


def get_answer_rules(mark_type: str | None) -> str:
    """Return base rules + mark-specific rules as one prompt block."""
    return f"{BASE_ANSWER_RULES}\n{get_mark_prompt(mark_type)}"


def target_word_range(mark_type: str | None) -> tuple[int, int]:
    """Return an approximate ``(min_words, max_words)`` band for a mark type."""
    marks = mark_int(mark_type)
    low = max(15, int(marks * 18 * 0.6))
    high = max(low + 30, int(marks * 26))
    return low, high


def marks_distribution(mark_type: str | None) -> str:
    """Human-readable distribution of a mark count, for UI captions."""
    marks = mark_int(mark_type)
    if marks <= 2:
        return "definition / short recall"
    if marks == 5:
        return "one paragraph, 3-4 points"
    if marks == 10:
        return "4-5 labelled sections + example"
    return "full model answer with diagram & applications"