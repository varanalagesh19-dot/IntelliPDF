"""MCQ quiz generation.

Two paths, same output contract:

* **LLM path** - retrieve chunks for the requested topic, ask Groq/Gemini/Ollama
  for a strict JSON array of MCQs, then parse it defensively (markdown fences,
  truncated arrays, stray prose).
* **Offline path** - when no provider is reachable, build MCQs deterministically
  from the retrieved sentences: one correct option (a real sentence) plus three
  plausible distractors taken from other sentences/pages. Useful for demos and
  tests, and honest about being heuristic.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from typing import Any

from backend.config import TOP_K
from backend.services.chunker import iter_text_fragments
from backend.services.llm_client import as_list_of_text, get_llm_client, parse_json_block
from backend.services.rag_pipeline import SYSTEM_ROLE, retrieve
from backend.services.vector_store import get_vector_store
from backend.utils import db
from backend.utils.metadata import chunk_id_for, iso_now

LOGGER = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")

#: Strips a leading option label such as ``A.``/``(B)``/``C:`` but never a
#: bare single-character option ("A" must survive as an answer choice).
_OPTION_LABEL_RE = re.compile(r"^\s*\(?[A-Da-d][\.\):]\s*|^\s*\([A-Da-d]\)\s*")

DIFFICULTY_GUIDANCE = {
    "easy": (
        "DIFFICULTY: easy - test direct recall of definitions, terms and "
        "single facts stated verbatim in the context."
    ),
    "medium": (
        "DIFFICULTY: medium - test understanding: 'Which of the following "
        "about X is correct?', comparisons, cause/effect, small inferences."
    ),
    "hard": (
        "DIFFICULTY: hard - test application and analysis: multi-step "
        "reasoning, 'Which statement is FALSE?', and why-style questions."
    ),
}

SYSTEM_ROLE_QUIZ = (
    "You are an exam question paper setter for an engineering student. You "
    "write only from the supplied context and you always reply with valid "
    "JSON - no markdown, no commentary."
)


def build_quiz_prompt(
    context: str, num_questions: int, difficulty: str, topic: str
) -> str:
    """Compose the MCQ generation prompt."""
    focus = topic.strip() or "the whole document"
    return (
        f"CONTEXT (use only this to write questions):\n{context}\n\n"
        f"Generate exactly {num_questions} multiple-choice questions about "
        f"{focus}.\n"
        f"{DIFFICULTY_GUIDANCE.get(difficulty, DIFFICULTY_GUIDANCE['medium'])}\n\n"
        "Rules:\n"
        "1. Exactly 4 options per question, labelled A-D in the 'options' array.\n"
        "2. Only one option is correct; distractors must be plausible but wrong "
        "and must NOT appear anywhere in the context.\n"
        "3. Every question must be answerable from the context alone.\n"
        "4. 'correct_index' is the 0-based index of the correct option.\n"
        "5. 'source_page' is the page number the answer came from.\n"
        "6. 'topic' is a short 2-4 word label for the concept tested.\n\n"
        'Reply with ONLY this JSON (no code fence):\n'
        '[{"question": "...", "options": ["...","...","...","..."], '
        '"correct_index": 0, "explanation": "...", "source_page": 3, '
        '"topic": "Vector Search", "difficulty": "medium"}]'
    )


def normalise_question(raw: dict[str, Any], default_topic: str) -> dict[str, Any] | None:
    """Validate and clean one LLM question object.

    Returns ``None`` when the object cannot be repaired into a usable MCQ.
    """
    question = str(raw.get("question") or raw.get("q") or "").strip()
    options = raw.get("options") or raw.get("choices") or []
    if isinstance(options, str):
        options = [opt for opt in re.split(r"[\n|;]+", options) if opt.strip()]

    cleaned: list[str] = []
    for option in list(options)[:4]:
        text = _OPTION_LABEL_RE.sub("", str(option)).strip()
        if text:
            cleaned.append(text)
    if not question or len(cleaned) < 2:
        return None

    while len(cleaned) < 4:  # pad short option lists so scoring stays uniform
        cleaned.append(f"None of the above (option {len(cleaned) + 1})")

    try:
        correct_index = int(raw.get("correct_index", raw.get("answer", 0)))
    except (TypeError, ValueError):
        correct_index = 0
    if raw.get("answer") is not None and isinstance(raw.get("answer"), str):
        letter = raw["answer"].strip().upper()[:1]
        if letter in LETTERS:
            correct_index = LETTERS.index(letter)
    correct_index = max(0, min(len(cleaned) - 1, correct_index))

    try:
        page = int(raw.get("source_page") or raw.get("page") or 0)
    except (TypeError, ValueError):
        page = 0

    return {
        "question": question,
        "options": cleaned,
        "correct_index": correct_index,
        "explanation": str(raw.get("explanation") or "").strip(),
        "source_page": page,
        "topic": str(raw.get("topic") or default_topic or "General").strip()[:40],
        "difficulty": str(raw.get("difficulty") or "medium").strip().lower(),
        "chunk_id": str(raw.get("chunk_id") or ""),
    }


def parse_quiz_json(raw: str, default_topic: str) -> list[dict[str, Any]]:
    """Parse the LLM reply into a list of clean questions."""
    payload = parse_json_block(raw)
    if isinstance(payload, dict):
        payload = payload.get("questions") or payload.get("quiz") or []
    if not isinstance(payload, list):
        return []

    questions: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = normalise_question(item, default_topic)
        if question:
            questions.append(question)
    return questions


# ── Offline generator ────────────────────────────────────────────────────


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_TERM_RE = re.compile(r"\b[A-Za-z][A-Za-z\-]{4,}\b")

#: Words that make poor quiz targets.
_GENERIC_TERMS = {
    "which", "there", "these", "those", "their", "about", "would", "should",
    "could", "using", "based", "called", "known", "given", "used", "make",
    "made", "such", "when", "where", "while", "after", "before", "because",
    "however", "therefore", "between", "through", "each", "other", "same",
    "different", "important", "example", "process", "system", "method",
}

DIFFICULTY_STEMS = {
    "easy": "From the material, complete this statement:",
    "medium": "Fill in the blank using the source material:",
    "hard": "A student wrote this in an exam. Which term completes it correctly?",
}


def _candidate_sentences(chunks: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    """Return ``(sentence, page, chunk_id)`` triples worth quizzing.

    Handles both prose (split on sentence boundaries) and table-like material
    (short lines re-joined into rows) so short technical PDFs still yield
    enough questions.
    """
    candidates: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        page = int(chunk.get("page_num", 0))
        chunk_id = chunk.get("chunk_id", "")
        for fragment in iter_text_fragments(chunk.get("text", "") or ""):
            for sentence in _SENTENCE_RE.split(fragment):
                sentence = sentence.strip(" .;|")
                if not (50 <= len(sentence) <= 340):
                    continue
                if len(sentence.split()) < 9 or not _TERM_RE.search(sentence):
                    continue
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((sentence, page, chunk_id))
    return candidates


def _salient_terms(sentences: list[tuple[str, int, str]]) -> list[str]:
    """Collect the distinctive vocabulary of the document, rarest first."""
    return _term_pool(_term_frequency(sentences), len(sentences))


def _term_frequency(sentences: list[tuple[str, int, str]]) -> Counter[str]:
    """Count how often each term appears across the candidate sentences."""
    frequency: Counter[str] = Counter()
    for sentence, _, _ in sentences:
        frequency.update(word.lower() for word in _TERM_RE.findall(sentence))
    return frequency


def _term_pool(frequency: Counter[str], sentence_count: int) -> list[str]:
    """Distinctive terms of the document, rarest first."""
    candidates = [
        term
        for term, count in frequency.items()
        if count <= max(3, sentence_count // 4)
        and term not in _GENERIC_TERMS
        and len(term) >= 4
    ]
    candidates.sort(key=lambda term: (frequency[term], -len(term)))
    return candidates


def _pick_target(
    sentence: str, frequency: Counter[str], used: set[str]
) -> str | None:
    """Choose which word of ``sentence`` to blank out.

    The *rarest* technical term wins - section headers usually come first in a
    line, so ties break towards the later position - which keeps the cloze
    focused on the concept rather than on the heading.
    """
    best: tuple[tuple[int, int], str] | None = None
    for position, word in enumerate(_TERM_RE.findall(sentence)):
        lowered = word.lower()
        if lowered in used or lowered in _GENERIC_TERMS or len(word) < 4:
            continue
        rank = (frequency.get(lowered, 0), position)
        if best is None or rank < best[0]:
            best = (rank, word)
    return best[1] if best else None


def offline_quiz(
    chunks: list[dict[str, Any]], num_questions: int, difficulty: str, topic: str
) -> list[dict[str, Any]]:
    """Build cloze-style MCQs (``term completion``) from retrieved sentences.

    Each question blanks out a distinctive term taken from the source; the
    distractors are other real terms from the same document, so the answer can
    only be found by reading the material.
    """
    sentences = _candidate_sentences(chunks)
    if not sentences:
        return []
    term_pool = _salient_terms(sentences)
    frequency = _term_frequency(sentences)
    if len(term_pool) < 4:
        return []

    stem = DIFFICULTY_STEMS.get(difficulty, DIFFICULTY_STEMS["medium"])
    questions: list[dict[str, Any]] = []
    used_terms: set[str] = set()
    used_questions: set[str] = set()

    for sentence, page, chunk_id in sentences:
        if len(questions) >= num_questions:
            break
        target = _pick_target(sentence, frequency, used_terms)
        if not target:
            continue
        blanked = re.sub(rf"\b{re.escape(target)}\b", "______", sentence, count=1)
        if blanked in used_questions:
            continue
        used_questions.add(blanked)
        used_terms.add(target.lower())

        # Distractors may repeat across questions, but never the answer.
        distractors = [
            term
            for term in term_pool
            if term.lower() != target.lower()
        ][:3]
        while len(distractors) < 3:
            distractors.append(f"None of the listed terms ({len(distractors) + 2})")

        options = [target, *distractors]
        topic_label = topic or "General"
        questions.append(
            {
                "question": f"{stem}\n\n{blanked}",
                "options": options,
                "correct_index": 0,
                "explanation": (
                    f"The missing term is **{target}** - it appears in this exact "
                    f"sentence on page {page}."
                ),
                "source_page": page,
                "topic": topic_label[:40],
                "difficulty": difficulty,
                "chunk_id": chunk_id,
            }
        )

    if not questions:  # nothing term-like: fall back to sentence spotting
        pool = [sentence for sentence, _, _ in sentences]
        for index in range(min(num_questions, len(pool))):
            sentence, page, chunk_id = sentences[index]
            options = [sentence] + [
                other for other in pool if other != sentence
            ][:3]
            questions.append(
                {
                    "question": "Which statement appears in the material?",
                    "options": options,
                    "correct_index": 0,
                    "explanation": f"Stated verbatim on page {page}.",
                    "source_page": page,
                    "topic": (topic or "General")[:40],
                    "difficulty": difficulty,
                    "chunk_id": chunk_id,
                }
            )
    return questions[:num_questions]


# ── Public API ───────────────────────────────────────────────────────────


def generate_quiz(
    doc_id: str,
    num_questions: int = 5,
    difficulty: str = "medium",
    topic: str = "",
    k: int = TOP_K,
) -> dict[str, Any]:
    """Generate an MCQ quiz for a document (optionally topic-scoped).

    Args:
        doc_id: Document identifier.
        num_questions: How many MCQs to produce (1-20).
        difficulty: ``easy`` / ``medium`` / ``hard``.
        topic: Optional topic focus; also filters retrieval.
        k: Chunks used as generation context.

    Returns:
        ``{"quiz_id", "questions", "provider", "warning"}``.

    Raises:
        FileNotFoundError: The document has no vector index.
    """
    store = get_vector_store()
    if not store.exists(doc_id):
        raise FileNotFoundError(f"No vector index for document {doc_id}")

    # A topic narrows retrieval; the question doubles as the query.
    query = topic.strip() or "key concepts definitions important terms"
    results = retrieve(doc_id, query, k=k)
    chunks = [result["chunk"] for result in results] or store.load(doc_id)["meta"][:k]

    context = "\n\n---\n\n".join(
        f"[Page {chunk['page_num']}]"
        + (f" — {chunk['heading']}" if chunk.get("heading") else "")
        + f"\n{chunk['text']}"
        for chunk in chunks
    )
    if not context.strip():
        raise ValueError("The document has no indexable text.")

    default_topic = (topic.strip() or "General")[:40]
    prompt = build_quiz_prompt(context, num_questions, difficulty, default_topic)
    response = get_llm_client().generate(prompt, system=SYSTEM_ROLE_QUIZ)

    questions: list[dict[str, Any]] = []
    warning = ""
    provider = "offline-heuristic"
    if response.ok:
        questions = parse_quiz_json(response.text, default_topic)
        provider = response.provider
        if not questions:
            warning = "The model reply was not valid JSON; used the offline generator."
    else:
        warning = "No LLM provider available; MCQs were generated offline from your PDF."

    if not questions:
        questions = offline_quiz(chunks, num_questions, difficulty, default_topic)

    # Stamp source pages from the retrieved chunks when the model omitted them.
    if not any(question.get("source_page") for question in questions):
        for index, question in enumerate(questions):
            question["source_page"] = int(chunks[index % len(chunks)].get("page_num", 0))
    for index, question in enumerate(questions):
        if not question.get("chunk_id"):
            chunk = chunks[index % len(chunks)]
            question["chunk_id"] = chunk.get(
                "chunk_id", chunk_id_for(doc_id, chunk.get("page_num", 0), index)
            )

    quiz_id = f"quiz_{uuid.uuid4().hex[:12]}"
    db.save_quiz(quiz_id, doc_id, questions, topic=topic, difficulty=difficulty)
    LOGGER.info(
        "Generated quiz %s for %s: %d questions via %s",
        quiz_id,
        doc_id,
        len(questions),
        provider,
    )
    return {
        "quiz_id": quiz_id,
        "doc_id": doc_id,
        "questions": questions,
        "topic": topic,
        "difficulty": difficulty,
        "provider": provider,
        "warning": warning,
        "created_at": iso_now(),
    }


def score_submission(
    doc_id: str | None = None,
    quiz_id: str | None = None,
    answers: list[int] | dict[Any, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score a quiz submission and persist the attempt.

    Args:
        doc_id: Document the quiz belongs to (optional, resolved from quiz if omitted).
        quiz_id: Stored quiz id.
        answers: Chosen option index per question (list or dict).

    Returns:
        ``{"score", "total", "percentage", "per_question_feedback",
        "weak_topics", "result_id"}``.

    Raises:
        FileNotFoundError: The quiz id is unknown.
    """
    # Support flexible positional/keyword calls:
    # 1. score_submission(quiz_id, answers)
    # 2. score_submission(doc_id, quiz_id, answers)
    # 3. score_submission(doc_id=..., quiz_id=..., answers=...)
    if isinstance(quiz_id, (list, dict)) and isinstance(doc_id, str):
        answers = quiz_id
        quiz_id = doc_id
        doc_id = None
    elif quiz_id is None and isinstance(doc_id, str) and not kwargs.get("quiz_id"):
        quiz_id = doc_id
        doc_id = None

    if not quiz_id and kwargs.get("quiz_id"):
        quiz_id = kwargs["quiz_id"]
    if answers is None and kwargs.get("answers") is not None:
        answers = kwargs["answers"]
    if not doc_id and kwargs.get("doc_id"):
        doc_id = kwargs["doc_id"]

    if not quiz_id:
        raise ValueError("A valid quiz_id is required to score submission.")

    quiz = db.get_quiz(quiz_id)
    if not quiz:
        raise FileNotFoundError(f"Unknown quiz: {quiz_id}")

    doc_id = str(doc_id or quiz.get("doc_id", ""))

    if answers is None:
        answers = []

    questions: list[dict[str, Any]] = quiz["questions"]
    feedback: list[dict[str, Any]] = []
    score = 0
    for index, question in enumerate(questions):
        try:
            if isinstance(answers, dict):
                selected = int(answers.get(index, answers.get(str(index), -1)))
            else:
                selected = int(answers[index])
        except (IndexError, TypeError, ValueError, KeyError):
            selected = -1
        is_correct = selected == int(question.get("correct_index", -1))
        score += int(is_correct)
        feedback.append(
            {
                "question": question.get("question", ""),
                "selected": selected,
                "correct_index": int(question.get("correct_index", -1)),
                "is_correct": is_correct,
                "explanation": question.get("explanation", ""),
                "source_page": int(question.get("source_page", 0)),
                "topic": question.get("topic", "General"),
                "options": question.get("options", []),
            }
        )

    total = len(questions)
    percentage = round(100.0 * score / total, 2) if total else 0.0

    # Weak topics come from the per-question feedback (post-aggregation).
    from backend.services.weak_topics import compute_topic_accuracy  # local import

    rows = compute_topic_accuracy(
        [{"details": feedback}]  # type: ignore[arg-type]
    )
    weak_topics = [row["topic"] for row in rows if row["accuracy"] < 0.6]

    result_id = f"res_{uuid.uuid4().hex[:12]}"
    db.save_quiz_result(
        result_id=result_id,
        doc_id=doc_id,
        quiz_id=quiz_id,
        score=score,
        total=total,
        weak_topics=weak_topics,
        details=feedback,
    )
    LOGGER.info("Scored %s: %d/%d (%.1f%%)", quiz_id, score, total, percentage)

    return {
        "quiz_id": quiz_id,
        "doc_id": doc_id,
        "result_id": result_id,
        "score": score,
        "total": total,
        "percentage": percentage,
        "per_question_feedback": feedback,
        "weak_topics": weak_topics,
    }


def quiz_topics(quiz: dict[str, Any]) -> list[str]:
    """Return the distinct topic labels used by a stored quiz."""
    seen: list[str] = []
    for question in quiz.get("questions", []):
        topic = question.get("topic") or "General"
        if topic not in seen:
            seen.append(topic)
    return seen


def explain_topics(topics: list[str]) -> str:
    """Render a topic list for prompts that need it inline."""
    return ", ".join(as_list_of_text(topics)) or "general syllabus"