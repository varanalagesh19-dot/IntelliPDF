"""Custom RAG pipeline: retrieve -> prompt -> generate -> ground in sources.

This is a deliberately explicit implementation of the pattern from Lewis et
al. (2020) so it can be explained and evaluated in a viva:

1. embed the question,
2. retrieve the top-k chunks from FAISS,
3. compose a prompt with ``[Page X]`` markers + mark rules + language rules,
4. call the LLM (Groq -> Gemini -> Ollama),
5. attach the retrieved chunks as verifiable sources.

If the best similarity is below :data:`config.SIMILARITY_THRESHOLD`, or no LLM
provider is reachable, the pipeline falls back to a deterministic *extractive*
answer assembled from the retrieved sentences, so the user always gets a
source-grounded response.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np

from backend.config import (
    HYBRID_DENSE_WEIGHT,
    HYBRID_LEXICAL_WEIGHT,
    INSUFFICIENT_INFO_ANSWER,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from backend.services.chunker import chunk_for_context, iter_text_fragments
from backend.services.embedder import get_embedder
from backend.services.llm_client import get_llm_client
from backend.services.mark_wise import (
    MARK_SENTENCE_BUDGET,
    get_answer_rules,
    mark_int,
    normalise_mark_type,
    target_word_range,
)
from backend.services.vector_store import get_vector_store
from backend.utils.language import get_language_instruction
from backend.utils.metadata import truncate

LOGGER = logging.getLogger(__name__)

SYSTEM_ROLE = (
    "You are IntelliPDF, an exam tutor for engineering students. You answer "
    "ONLY from the provided context, you never invent facts, and you always "
    "respect the requested mark-wise length and language."
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on",
    "for", "to", "and", "or", "what", "which", "who", "how", "why", "when",
    "where", "define", "definition", "explain", "describe", "about", "with",
    "from", "that", "this", "these", "those", "it", "its", "as", "by", "at",
    "can", "will", "any", "give", "write", "list", "state", "mention", "briefly",
}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9]+", (text or "").lower()) if t]


def _content_terms(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in STOPWORDS and len(t) > 2}


def _split_sentences(text: str) -> list[str]:
    """Return answer-ready sentence fragments from a chunk.

    Table-like PDFs store each cell on its own line with no sentence
    punctuation, so the raw chunk would be one enormous "sentence".  The text
    is first reconstructed into row-level fragments (see
    :func:`backend.services.chunker.iter_text_fragments`) and only then split
    on sentence boundaries.
    """
    sentences: list[str] = []
    for fragment in iter_text_fragments(text or ""):
        for sentence in _SENTENCE_RE.split(fragment):
            sentence = sentence.strip()
            if len(sentence) > 30:
                sentences.append(sentence)
    return sentences


def _condense(sentence: str, max_words: int = 70) -> list[str]:
    """Split an over-long sentence into clause-sized answer fragments."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    clauses = re.split(r"(?<=,)\s+|\s+(?:and|but|while|which|where)\s+", sentence)
    pieces: list[str] = []
    for clause in clauses:
        pieces.extend(
            " ".join(part.split()) for part in _wrap(clause, max_words)
        )
    return [piece for piece in pieces if len(piece) > 25] or [sentence]


def _wrap(text: str, max_words: int) -> list[str]:
    """Split ``text`` into consecutive groups of at most ``max_words`` words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# ── Hybrid retrieval ─────────────────────────────────────────────────────
# MiniLM cosine similarity alone under-scores short documents (with only a
# handful of chunks the query has little to compete against) and misses exact
# terminology matches such as "chunks -> numerical vectors".  Adding an
# IDF-weighted lexical score recovers those cases while the dense score keeps
# the semantic matching, which is the classic hybrid-retrieval setup.


def _lexical_scores(query: str, chunks: list[dict[str, Any]]) -> np.ndarray:
    """Return IDF-weighted query-coverage scores for every chunk."""
    scores = np.zeros(len(chunks), dtype=np.float32)
    if not chunks:
        return scores
    chunk_tokens = [set(_tokens(chunk.get("text", ""))) for chunk in chunks]
    total = len(chunk_tokens)
    document_frequency: dict[str, int] = {}
    for tokens in chunk_tokens:
        for token in tokens:
            if token in STOPWORDS or len(token) < 2:
                continue
            document_frequency[token] = document_frequency.get(token, 0) + 1

    query_terms = [t for t in set(_tokens(query)) if t not in STOPWORDS and len(t) > 1]
    if not query_terms:
        return scores
    weights = {
        token: math.log((total + 1) / (document_frequency.get(token, 0) + 1)) + 1.0
        for token in query_terms
    }
    denominator = sum(weights.values()) or 1.0
    for index, tokens in enumerate(chunk_tokens):
        hits = sum(weight for token, weight in weights.items() if token in tokens)
        scores[index] = hits / denominator
    return scores


def hybrid_rank(
    doc_id: str,
    query: str,
    k: int = TOP_K,
    page_filter: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks with a weighted dense + lexical score.

    Args:
        doc_id: Document identifier.
        query: Natural-language question.
        k: Number of chunks to return.
        page_filter: Optional page restriction.

    Returns:
        ``[{"chunk", "score", "dense", "lexical", "rank"}]`` sorted by the
        combined score.
    """
    store = get_vector_store()
    dense_results = store.search(doc_id, get_embedder().encode_query(query), k=k,
                                 page_filter=page_filter)
    if not dense_results:
        return []

    meta = store.load(doc_id)["meta"]
    indices = [meta.index(dense["chunk"]) for dense in dense_results]
    lexical = _lexical_scores(query, [dense["chunk"] for dense in dense_results])

    combined: list[dict[str, Any]] = []
    for position, dense in enumerate(dense_results):
        dense_score = max(0.0, float(dense["score"]))
        lex_score = float(lexical[position])
        combined.append(
            {
                "chunk": dense["chunk"],
                "dense": round(dense_score, 4),
                "lexical": round(lex_score, 4),
                "score": round(
                    HYBRID_DENSE_WEIGHT * dense_score + HYBRID_LEXICAL_WEIGHT * lex_score, 4
                ),
            }
        )
    combined.sort(key=lambda item: -item["score"])
    for rank, item in enumerate(combined, start=1):
        item["rank"] = rank
    LOGGER.debug("Hybrid retrieval indices=%s top=%.3f", indices[:3], combined[0]["score"])
    return combined[:k]


# ── Retrieval ────────────────────────────────────────────────────────────


def retrieve(
    doc_id: str, query: str, k: int = TOP_K, page_filter: set[int] | None = None
) -> list[dict[str, Any]]:
    """Embed ``query`` and return the ``k`` best chunks for ``doc_id``.

    Uses the hybrid (dense + lexical) ranker; ``page_filter`` restricts the
    search to specific pages.

    Args:
        doc_id: Document identifier with a stored index.
        query: Natural-language question.
        k: Number of chunks to retrieve.
        page_filter: Optional page numbers to restrict retrieval to.

    Returns:
        ``[{"chunk", "score", "dense", "lexical", "rank"}]`` ordered by
        descending score.
    """
    return hybrid_rank(doc_id, query, k=k, page_filter=page_filter)


def build_context(results: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as a context block with page markers."""
    blocks: list[str] = []
    for result in results:
        chunk = result["chunk"]
        blocks.append(
            f"{chunk_for_context(chunk)}\n(source: {chunk['chunk_id']}, "
            f"similarity {result['score']:.2f})"
        )
    return "\n\n---\n\n".join(blocks)


def confidence_from_scores(results: list[dict[str, Any]]) -> tuple[float, str]:
    """Return ``(score, label)`` summarising retrieval confidence.

    The score is the mean of the top-3 similarities; the label is
    ``high`` (>= 0.55), ``medium`` (>= 0.35) or ``low``.
    """
    if not results:
        return 0.0, "low"
    top = [result["score"] for result in results[:3]]
    score = round(sum(top) / len(top), 4)
    if score >= 0.55:
        label = "high"
    elif score >= 0.35:
        label = "medium"
    else:
        label = "low"
    return score, label


def _sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert retrieval results into UI source chips."""
    seen: set[int] = set()
    sources: list[dict[str, Any]] = []
    for result in results:
        chunk = result["chunk"]
        page = int(chunk.get("page_num", 0))
        if page in seen:
            continue
        seen.add(page)
        sources.append(
            {
                "page": page,
                "chunk_id": chunk.get("chunk_id", ""),
                "heading": chunk.get("heading") or "",
                "snippet": truncate(chunk.get("text", ""), 320),
                "score": round(float(result["score"]), 4),
            }
        )
    return sources


# ── Prompting ────────────────────────────────────────────────────────────


def build_prompt(
    question: str,
    context: str,
    mark_type: str,
    language: str,
    extra_instructions: str = "",
) -> str:
    """Compose the full user prompt for one RAG call."""
    marks = mark_int(mark_type)
    parts = [
        f"CONTEXT FROM THE STUDENT'S PDF (use only this):\n{context}",
        get_answer_rules(mark_type),
        get_language_instruction(language),
        f"FORMAT: The answer is worth {marks} mark"
        f"{'s' if marks != 1 else ''}. Follow the mark rule exactly.",
    ]
    if extra_instructions:
        parts.append(extra_instructions)
    parts.append(f"STUDENT QUESTION:\n{question.strip()}")
    return "\n\n".join(parts)


# ── Offline extractive fallback ──────────────────────────────────────────


_WHEN_RE = re.compile(
    r"\b(what|which)\s+year\b|\bin\s+what\s+year\b|\bwhen\b|\bwhat\s+date\b", re.I
)
_HOWMANY_RE = re.compile(r"\bhow\s+(many|much|long|old|far|tall)\b", re.I)
_WHO_RE = re.compile(r"\bwho(m|se)?\b", re.I)
_NAMED_RE = re.compile(
    r"\bwhat\s+(is|are|was|were)\b.*\b(called|named|known|defined)\b", re.I
)
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")


def expected_answer_type(question: str) -> str:
    """Classify the answer a question expects: number / person / proper / any."""
    text = question or ""
    if _WHEN_RE.search(text) or _HOWMANY_RE.search(text):
        return "number"
    if _WHO_RE.search(text):
        return "person"
    if _NAMED_RE.search(text):
        return "proper"
    return "any"


def answer_type_supported(question: str, sentence: str) -> bool:
    """Check that the retrieved sentence contains something of the asked type.

    Retrieval confidence alone cannot detect an unanswerable question when the
    whole document is a single chunk (there is nothing to compare against), so
    the offline path additionally verifies that the evidence could plausibly
    contain a number, a name, ... - whatever the question asks for.  With an LLM
    configured this check is bypassed: the model decides for itself.
    """
    kind = expected_answer_type(question)
    if kind == "number":
        return bool(_NUMBER_RE.search(sentence))
    if kind in {"person", "proper"}:
        question_words = set(_tokens(question))
        candidates = {
            word
            for word in _PROPER_RE.findall(sentence)
            if word.lower() not in question_words
        }
        return bool(candidates)
    return True


def _shortest_relevant_clause(sentence: str, query_terms: set[str]) -> str:
    """Trim a sentence down to the clause that actually answers the question.

    A 1M/2M answer must be one precise line, so the sentence is split on
    commas/conjunctions and the clause with the highest keyword density wins.
    """
    clauses = [
        clause.strip(" ,;:")
        for clause in re.split(r",\s+|\s+(?:and|but|while|which|who|where)\s+", sentence)
        if len(clause.strip(" ,;:")) > 15
    ]
    if len(clauses) < 2:
        return sentence
    scored = sorted(
        clauses,
        key=lambda clause: (
            -len(_content_terms(clause) & query_terms) / max(1, len(_content_terms(clause))),
            len(clause.split()),
        ),
    )
    return scored[0]


def extractive_answer(
    question: str,
    results: list[dict[str, Any]],
    mark_type: str = "5M",
    language: str = "english",
) -> str:
    """Build a deterministic answer from the retrieved chunks.

    Used when no LLM provider is reachable.  Sentences are ranked by keyword
    overlap with the question and by retrieval score, then laid out following
    the mark-wise shape so the offline answer still looks like a model answer.
    """
    if not results:
        return INSUFFICIENT_INFO_ANSWER

    marks = mark_int(mark_type)
    _low_words, high_words = target_word_range(mark_type)
    query_terms = _content_terms(question)

    scored: list[tuple[float, int, str]] = []
    for result in results:
        chunk = result["chunk"]
        for position, raw in enumerate(_split_sentences(chunk["text"])):
            for sentence in _condense(raw):
                terms = _content_terms(sentence)
                if not terms:
                    continue
                overlap = len(terms & query_terms)
                coverage = overlap / max(1, len(query_terms))
                scored.append(
                    (coverage + float(result["score"]) * 0.5, position, sentence)
                )

    if not scored:
        best = results[0]["chunk"]
        return truncate(best["text"], high_words)

    scored.sort(key=lambda item: (-item[0], item[1]))
    budget_sentences = MARK_SENTENCE_BUDGET.get(normalise_mark_type(mark_type), 5)
    chosen: list[str] = []
    seen_terms: set[str] = set()
    used_words = 0
    for _score, _position, sentence in scored:
        words = len(sentence.split())
        terms = _content_terms(sentence)
        if chosen and terms and len(terms & seen_terms) / max(1, len(terms)) > 0.7:
            continue  # near-duplicate sentence
        chosen.append(sentence.strip())
        seen_terms |= terms
        used_words += words
        if used_words >= high_words or len(chosen) >= budget_sentences:
            break

    if not chosen:
        chosen = [results[0]["chunk"]["text"][:400]]

    # The evidence must be able to hold the kind of answer that was asked for.
    evidence = " ".join(chosen[:2])
    if not answer_type_supported(question, evidence):
        LOGGER.info(
            "Offline abstention: %r expects a %s but the retrieved text has none",
            question,
            expected_answer_type(question),
        )
        return INSUFFICIENT_INFO_ANSWER

    if marks <= 2:
        answer = _shortest_relevant_clause(chosen[0], query_terms)
        if marks == 2 and len(chosen) > 1:
            second = _shortest_relevant_clause(chosen[1], query_terms)
            if second.lower() != answer.lower():
                answer = f"{answer} - {second}"
    elif marks == 5:
        answer = "\n".join(f"- {sentence}" for sentence in chosen[:5])
    else:
        heading = results[0]["chunk"].get("heading") or question.strip()
        sections = "\n\n".join(
            f"## Point {i + 1}\n{sentence}" for i, sentence in enumerate(chosen[:6])
        )
        answer = (
            f"### {heading}\n\n{sections}\n\n"
            "*(Generated offline from your PDF because no LLM API key was "
            "reachable - add GROQ_API_KEY for full model answers.)*"
        )
    return answer


# ── Public API ───────────────────────────────────────────────────────────


def rag_query(
    doc_id: str,
    query: str,
    mark_type: str = "5M",
    language: str = "english",
    k: int = TOP_K,
    extra_instructions: str = "",
) -> dict[str, Any]:
    """Answer a question about a document, grounded in retrieved chunks.

    Args:
        doc_id: Document identifier.
        query: The student's question.
        mark_type: ``1M`` / ``2M`` / ``5M`` / ``10M`` / ``15M`` / ``16M``.
        language: ``english`` / ``tamil`` / ``tanglish``.
        k: Chunks retrieved before generation.
        extra_instructions: Optional extra prompt guidance.

    Returns:
        ``{"answer", "sources", "confidence", "confidence_label", "provider",
        "mark_type", "language", "grounded"}``.

    Raises:
        ValueError: The query is empty or the document index is missing.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Question cannot be empty.")
    if len(query) > 1000:
        query = query[:1000]

    mark_type = normalise_mark_type(mark_type)
    results = retrieve(doc_id, query, k=k)
    confidence, label = confidence_from_scores(results)
    top_score = results[0]["score"] if results else 0.0

    if not results or top_score < SIMILARITY_THRESHOLD:
        LOGGER.info(
            "Low-confidence retrieval for %r (top=%.3f) -> abstaining", query, top_score
        )
        return {
            "answer": INSUFFICIENT_INFO_ANSWER,
            "sources": _sources(results),
            "confidence": round(top_score, 4),
            "confidence_label": "low",
            "provider": "abstain",
            "mark_type": mark_type,
            "language": language,
            "grounded": False,
        }

    context = build_context(results)
    prompt = build_prompt(query, context, mark_type, language, extra_instructions)
    response = get_llm_client().generate(prompt, system=SYSTEM_ROLE)

    if response.ok:
        answer = response.text.strip()
        provider = response.provider
    else:
        answer = extractive_answer(query, results, mark_type, language)
        provider = "extractive-fallback"

    return {
        "answer": answer,
        "sources": _sources(results),
        "confidence": confidence,
        "confidence_label": label,
        "provider": provider,
        "model": getattr(response, "model", ""),
        "mark_type": mark_type,
        "language": language,
        "grounded": True,
    }


def answer_from_chunks(
    doc_id: str,
    question: str,
    chunks: list[dict[str, Any]],
    mark_type: str = "5M",
    language: str = "english",
) -> str:
    """Run one RAG answer restricted to a fixed set of chunks.

    Used by quiz generation and revision cards, which already know which pages
    or topics they want to talk about.

    Args:
        doc_id: Document identifier.
        question: Instruction such as "Define RAG".
        chunks: Chunks (with ``score``/``rank`` optional) to ground the answer.
        mark_type: Mark-wise rules to follow.
        language: Output language.

    Returns:
        The generated (or extractive) answer text.
    """
    if not chunks:
        return INSUFFICIENT_INFO_ANSWER
    normalised = [
        {"chunk": chunk, "score": chunk.get("score", 0.5), "rank": index}
        for index, chunk in enumerate(chunks, start=1)
    ]
    prompt = build_prompt(
        question, build_context(normalised), mark_type, language
    )
    response = get_llm_client().generate(prompt, system=SYSTEM_ROLE)
    if response.ok:
        return response.text.strip()
    return extractive_answer(question, normalised, mark_type, language)