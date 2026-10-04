"""Language support for English, Tamil and Tanglish.

The UI is multilingual (students in Tamil Nadu frequently study English
material); the *question* is usually asked in English or Tanglish while the
*answer* may be requested in Tamil.  Every LLM prompt therefore gets a short
language instruction injected, and :func:`normalize_language` guarantees a
safe value even when the frontend sends an unexpected string.
"""

from __future__ import annotations

import logging
from typing import Final

LOGGER = logging.getLogger(__name__)

ENGLISH: Final[str] = "english"
TAMIL: Final[str] = "tamil"
TANGLISH: Final[str] = "tanglish"

SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = (ENGLISH, TAMIL, TANGLISH)

#: Human readable labels for the Streamlit selector.
LANGUAGE_LABELS: Final[dict[str, str]] = {
    ENGLISH: "English",
    TAMIL: "Tamil (தமிழ்)",
    TANGLISH: "Tanglish (Tamil + English)",
}

#: Instruction appended to every prompt so the model answers in the right
#: language.  Technical terms are always kept in English to stay searchable.
LANG_INSTRUCTIONS: Final[dict[str, str]] = {
    ENGLISH: (
        "LANGUAGE RULE: Respond in clear, simple English. Keep standard "
        "technical terminology in English. Be concise and exam-oriented."
    ),
    TAMIL: (
        "LANGUAGE RULE: Respond in Tamil (தமிழ்). Use simple Tamil suitable "
        "for students, but keep unavoidable technical terms (RAG, embedding, "
        "FAISS, transformer...) in English. Answer as a friendly Indian "
        "professor would to a student."
    ),
    TANGLISH: (
        "LANGUAGE RULE: Respond in Tanglish — Tamil words written in the "
        "English script, mixed with English technical terms. "
        "Example: 'RAG oru retrieval technique, idhu LLM kooda work aagum. "
        "Document ellaa chunk aaga split aagum, mela embedding vector-aaga "
        "store aagum, epdi search aagum-nu sollunga.' Keep the tone friendly "
        "and exam-focused."
    ),
}

DEFAULT_LANGUAGE: Final[str] = ENGLISH


def normalize_language(language: str | None) -> str:
    """Coerce arbitrary user input into a supported language key.

    Args:
        language: Raw language string from an API payload or the UI.

    Returns:
        One of ``english``, ``tamil`` or ``tanglish`` (defaults to English).
    """
    if not language:
        return DEFAULT_LANGUAGE
    value = str(language).strip().lower()
    if value in LANG_INSTRUCTIONS:
        return value
    aliases = {
        "en": ENGLISH,
        "eng": ENGLISH,
        "hi-in": ENGLISH,
        "ta": TAMIL,
        "tam": TAMIL,
        "tang": TANGLISH,
        "tanglish": TANGLISH,
        "tamil_english": TANGLISH,
    }
    if value in aliases:
        return aliases[value]
    LOGGER.warning("Unsupported language %r, defaulting to English", language)
    return DEFAULT_LANGUAGE


def get_language_instruction(language: str | None) -> str:
    """Return the prompt instruction for ``language`` (always non-empty)."""
    return LANG_INSTRUCTIONS[normalize_language(language)]


def is_tamil_script(text: str) -> bool:
    """Return ``True`` when the Tamil Unicode block is present in ``text``."""
    return any("\u0b80" <= ch <= "\u0bff" for ch in text or "")