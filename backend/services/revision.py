"""Last-minute revision: definitions, key concepts and quick-fire flashcards.

Cards are generated from chunks sampled across the whole document (not just
the first pages) so a 10-card revision set covers the entire syllabus.  The
LLM path asks for strict JSON; the offline path mines definitional sentences
("X is ...") and high-centrality sentences with a small TF-IDF style score.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from backend.services.chunker import iter_text_fragments
from backend.services.llm_client import get_llm_client, parse_json_block
from backend.services.rag_pipeline import SYSTEM_ROLE, build_context
from backend.services.vector_store import get_vector_store
from backend.utils.language import get_language_instruction

LOGGER = logging.getLogger(__name__)

SYSTEM_ROLE_REVISION = (
    "You are a revision coach for the night before an exam. You write short, "
    "high-yield flashcards strictly from the supplied material and you always "
    "reply with valid JSON only."
)

MODE_GUIDANCE = {
    "last_minute": (
        "Mix everything: the 8 definitions that are asked most often, 2 key "
        "concepts and 2 formulas/terminology. Keep each card under 25 words."
    ),
    "definitions": (
        "Every card must be a definition card: front = the term, back = a "
        "precise 1-2 sentence definition taken from the context. 10-18 words."
    ),
    "key_concepts": (
        "Every card must test a concept: front = 'Why/How/What is the role "
        "of ...', back = 2-3 bullet points that explain the concept in the "
        "student's own exam-ready words, grounded in the context."
    ),
}

_DEF_PATTERNS = (
    re.compile(r"^(?P<term>[A-Z][A-Za-z0-9 ,()\-/]{2,60}?)\s+(?:is|are|was|were)\s+(?P<body>.{20,300})"),
    re.compile(r"^(?P<term>[A-Z][A-Za-z0-9 ,()\-/]{2,60}?)\s*(?::|,)\s*(?P<body>.{20,300})"),
    re.compile(r"^(?P<term>.{2,60}?)\s+(?:refers to|means|is defined as|is called)\s+(?P<body>.{20,300})"),
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "are", "was", "were", "from",
    "which", "have", "has", "been", "their", "there", "these", "those", "when",
    "then", "than", "into", "also", "used", "using", "such", "other", "more",
    "most", "can", "will", "would", "should", "could", "each", "any", "all",
}


def sample_chunks(doc_id: str, num_cards: int) -> list[dict[str, Any]]:
    """Pick chunks spread evenly across the document for maximum coverage."""
    meta = get_vector_store().load(doc_id)["meta"]
    if not meta:
        return []
    wanted = max(num_cards * 2, 6)
    if len(meta) <= wanted:
        return list(meta)
    step = len(meta) / wanted
    return [meta[int(index * step)] for index in range(wanted)]


def _central_sentences(chunks: list[dict[str, Any]], limit: int) -> list[tuple[str, int]]:
    """Rank sentences by term frequency across the sampled chunks."""
    frequencies: Counter[str] = Counter()
    sentences: list[tuple[str, int]] = []
    for chunk in chunks:
        for fragment in iter_text_fragments(chunk.get("text", "")):
            for sentence in _SENTENCE_RE.split(fragment):
                sentence = sentence.strip()
                if 45 <= len(sentence) <= 400:
                    sentences.append((sentence, int(chunk["page_num"])))
                    frequencies.update(
                        word.lower() for word in _WORD_RE.findall(sentence)
                    )
    scored = [
        (sentence, page, sum(frequencies[word.lower()] for word in _WORD_RE.findall(sentence)))
        for sentence, page in sentences
    ]
    scored.sort(key=lambda item: -item[2])
    return [(sentence, page) for sentence, page, _ in scored[:limit]]


def _definition_cards(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Mine ``X is ...`` style sentences into definition cards."""
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        for fragment in iter_text_fragments(chunk.get("text", "")):
            for sentence in _SENTENCE_RE.split(fragment):
                card = _definition_from_sentence(sentence, chunk["page_num"], seen)
                if card:
                    cards.append(card)
                    seen.add(card["front"].removeprefix("Define: ").lower())
                if len(cards) >= limit:
                    return cards
    return cards


def _definition_from_sentence(
    sentence: str, page: int, seen: set[str]
) -> dict[str, Any] | None:
    """Return a definition card if ``sentence`` matches a definitional pattern."""
    sentence = sentence.strip()
    for pattern in _DEF_PATTERNS:
        match = pattern.match(sentence)
        if not match:
            continue
        term = match.group("term").strip(" ,:")
        body = match.group("body").strip()
        if len(term) < 3 or term.lower() in seen:
            return None
        return {
            "front": f"Define: {term}",
            "back": body if body.endswith((".", "!", "?")) else body + ".",
            "source_page": int(page),
        }
    return None


def offline_cards(
    chunks: list[dict[str, Any]], mode: str, num_cards: int
) -> list[dict[str, Any]]:
    """Build revision cards without any LLM provider."""
    if mode == "definitions":
        cards = _definition_cards(chunks, num_cards)
        if len(cards) >= num_cards:
            return cards
        # Too few definitional sentences - top up with concept cards so the
        # student still gets a full revision set.
        return (cards + _concept_cards(chunks, num_cards, cards, mode))[:num_cards]

    return _concept_cards(chunks, num_cards, [], mode)[:num_cards]


def _concept_cards(
    chunks: list[dict[str, Any]],
    num_cards: int,
    existing: list[dict[str, Any]],
    mode: str = "last_minute",
) -> list[dict[str, Any]]:
    """Build "recall / key idea" cards from central sentences."""
    already = {card["back"][:40].lower() for card in existing}
    cards: list[dict[str, Any]] = []
    seen_fronts: set[str] = set()
    for sentence, page in _central_sentences(chunks, num_cards * 3):
        words = _WORD_RE.findall(sentence)
        if len(words) < 6 or sentence[:40].lower() in already:
            continue
        front = " ".join(words[:6]).strip(" ,;:")
        if front.lower() in seen_fronts:
            continue
        seen_fronts.add(front.lower())
        cards.append(
            {
                "front": f"{'Key idea' if mode == 'key_concepts' else 'Recall'}: {front}...",
                "back": sentence,
                "source_page": page,
            }
        )
        if len(cards) >= num_cards:
            break

    if not cards:  # nothing sentence-like: fall back to chunk excerpts
        for chunk in chunks[:num_cards]:
            cards.append(
                {
                    "front": (chunk.get("heading") or f"Page {chunk['page_num']}")[:80],
                    "back": chunk["text"][:320],
                    "source_page": int(chunk["page_num"]),
                }
            )
    return cards[:num_cards]


def build_revision_prompt(
    context: str, mode: str, num_cards: int, language: str
) -> str:
    """Compose the flashcard prompt."""
    return (
        f"MATERIAL:\n{context}\n\n"
        f"Create {num_cards} revision flashcards.\n{MODE_GUIDANCE[mode]}\n"
        "Every card must be answerable from the material above and must name the "
        "page it came from.\n"
        f"{get_language_instruction(language)}\n\n"
        'Reply with ONLY this JSON (no code fence):\n'
        '[{"front": "question or term", "back": "concise answer", '
        '"source_page": 4}]'
    )


def generate_revision_cards(
    doc_id: str, mode: str = "last_minute", num_cards: int = 10, language: str = "english"
) -> dict[str, Any]:
    """Generate revision cards for a document.

    Args:
        doc_id: Document identifier.
        mode: ``last_minute`` / ``definitions`` / ``key_concepts``.
        num_cards: How many cards to return.
        language: Output language.

    Returns:
        ``{"cards": [{"front", "back", "source_page"}], "provider", "sources"}``.

    Raises:
        FileNotFoundError: The document has no vector index.
    """
    mode = mode if mode in MODE_GUIDANCE else "last_minute"
    chunks = sample_chunks(doc_id, num_cards)
    if not chunks:
        raise FileNotFoundError(f"No content indexed for document {doc_id}")

    context = build_context([{"chunk": chunk, "score": 1.0, "rank": i + 1}
                             for i, chunk in enumerate(chunks)])
    prompt = build_revision_prompt(context, mode, num_cards, language)
    response = get_llm_client().generate(
        prompt, system=SYSTEM_ROLE_REVISION, max_tokens=2000
    )

    cards: list[dict[str, Any]] = []
    provider = "extractive-fallback"
    warning = ""
    if response.ok:
        payload = parse_json_block(response.text)
        if isinstance(payload, dict):
            payload = payload.get("cards") or payload.get("flashcards") or []
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                front = str(item.get("front") or item.get("question") or "").strip()
                back = str(item.get("back") or item.get("answer") or "").strip()
                if not front or not back:
                    continue
                try:
                    page = int(item.get("source_page") or item.get("page") or 0)
                except (TypeError, ValueError):
                    page = 0
                cards.append({"front": front, "back": back, "source_page": page})
            provider = response.provider

    if not cards:
        cards = offline_cards(chunks, mode, num_cards)
        warning = "Cards built offline from your PDF (no LLM key reachable)."

    pages = sorted({card["source_page"] for card in cards if card["source_page"]})
    LOGGER.info("Revision cards for %s (%s): %d via %s", doc_id, mode, len(cards), provider)

    sources = [
        {
            "page": page,
            "chunk_id": "",
            "heading": "",
            "snippet": "",
            "score": 1.0,
        }
        for page in pages
    ]
    return {
        "doc_id": doc_id,
        "mode": mode,
        "cards": cards[:num_cards],
        "provider": provider,
        "warning": warning,
        "sources": sources,
    }