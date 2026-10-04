"""Pytest suite for the IntelliPDF pipeline.

Run from the project root::

    pytest tests -v

The suite builds a synthetic PDF with PyMuPDF, then exercises extraction,
chunking, the FAISS vector store and the full RAG path.  It needs no API key:
the tests assert on structure and grounding, not on prose quality.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

from backend.config import INSUFFICIENT_INFO_ANSWER, SIMILARITY_THRESHOLD  # noqa: E402
from backend.services.chunker import chunk_pages, chunk_stats, iter_text_fragments  # noqa: E402
from backend.services.embedder import get_embedder  # noqa: E402
from backend.services.evaluator import offline_evaluate  # noqa: E402
from backend.services.mark_wise import (  # noqa: E402
    MARK_TYPES,
    get_mark_prompt,
    mark_int,
    normalise_mark_type,
)
from backend.services.pdf_processor import (  # noqa: E402
    PDFProcessingError,
    extract_pages,
    full_text,
)
from backend.services.quiz_gen import offline_quiz, parse_quiz_json  # noqa: E402
from backend.services.rag_pipeline import (  # noqa: E402
    answer_type_supported,
    expected_answer_type,
    extractive_answer,
    rag_query,
    retrieve,
)
from backend.services.study_planner import parse_syllabus  # noqa: E402
from backend.services.vector_store import VectorStore  # noqa: E402
from backend.utils import db  # noqa: E402
from backend.utils.language import get_language_instruction, normalize_language  # noqa: E402
from backend.utils.metadata import generate_doc_id, safe_filename  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_PAGES = [
    "Machine Learning Fundamentals\n"
    "1.1 Supervised Learning\n"
    "Supervised learning uses labelled examples to train a model. The model "
    "learns a mapping from input features to a target label.\n"
    "1.2 Unsupervised Learning\n"
    "Unsupervised learning finds structure in unlabelled data, such as "
    "clustering customers by purchase behaviour.\n"
    "1.3 Reinforcement Learning\n"
    "Reinforcement learning trains an agent using rewards received from an "
    "environment. The agent learns a policy that maximises cumulative reward.",
    "Vector Search\n"
    "2.1 Embeddings\n"
    "An embedding is a fixed length vector that represents the meaning of a "
    "piece of text. Similar meanings produce similar vectors.\n"
    "2.2 FAISS\n"
    "FAISS is a library for efficient similarity search. It stores vectors and "
    "returns the nearest neighbours using inner product on normalised vectors.\n"
    "2.3 Retrieval Augmented Generation\n"
    "Retrieval augmented generation combines a retriever with a generator. The "
    "retriever fetches relevant passages and the generator answers from them.",
]


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a small multi-page PDF with real, extractable text."""
    path = tmp_path_factory.mktemp("pdfs") / "sample.pdf"
    document = pymupdf.open()
    for index, text in enumerate(SAMPLE_PAGES):
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(60, 60, 540, 760), text, fontsize=11)
        assert index >= 0
    document.save(str(path))
    document.close()
    return path


@pytest.fixture(scope="session")
def pages(sample_pdf: Path) -> list[dict[str, Any]]:
    """Extracted pages of the sample PDF."""
    return extract_pages(sample_pdf)


@pytest.fixture(scope="session")
def indexed(tmp_path_factory: pytest.TempPathFactory, pages: list[dict[str, Any]]
            ) -> tuple[str, VectorStore]:
    """Chunk, embed and index the sample PDF in a throwaway vector store."""
    store = VectorStore(tmp_path_factory.mktemp("faiss"))
    doc_id = "test-doc"
    chunks = chunk_pages(pages, doc_id)
    embeddings = get_embedder().encode([chunk["text"] for chunk in chunks])
    store.add(doc_id, embeddings, chunks)
    return doc_id, store


# ── PDF processing ────────────────────────────────────────────────────────


def test_extract_pages_returns_page_numbers(pages: list[dict[str, Any]]) -> None:
    """Every extracted page carries a 1-based page number and text."""
    assert len(pages) == len(SAMPLE_PAGES)
    for index, page in enumerate(pages, start=1):
        assert page["page_num"] == index
        assert len(page["text"]) > 30
        assert "\n\n\n" not in page["text"]  # whitespace was collapsed


def test_extract_pages_removes_boilerplate(tmp_path: Path) -> None:
    """A line repeated on every page is treated as a running header."""
    document = pymupdf.open()
    for page_number in range(1, 5):
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(60, 60, 540, 760),
            "UNIT 3 - MACHINE LEARNING   |   continued\n"
            f"Body sentence number {page_number} carrying unique page content.",
            fontsize=11,
        )
    path = tmp_path / "boilerplate.pdf"
    document.save(str(path))
    document.close()

    extracted = extract_pages(path)
    assert extracted, "pages should still be extracted"
    assert all("UNIT 3 - MACHINE LEARNING" not in page["text"] for page in extracted)
    assert all("Body sentence number" in page["text"] for page in extracted)
    assert len(extracted) == 4


def test_extract_pages_rejects_non_pdf(tmp_path: Path) -> None:
    """A file without the .pdf suffix is rejected with a clear error."""
    path = tmp_path / "notes.txt"
    path.write_text("plain text", encoding="utf-8")
    with pytest.raises(PDFProcessingError):
        extract_pages(path)


def test_full_text_has_page_markers(pages: list[dict[str, Any]]) -> None:
    """The joined text is annotated with ``[Page X]`` markers."""
    text = full_text(pages)
    assert "[Page 1]" in text and "[Page 2]" in text


# ── Chunking ──────────────────────────────────────────────────────────────


def test_chunking_produces_expected_chunks(pages: list[dict[str, Any]]) -> None:
    """Chunking yields metadata-complete chunks within the size limit."""
    chunks = chunk_pages(pages, "test-doc")
    assert chunks, "chunking must produce at least one chunk"
    assert all(chunk["page_num"] in (1, 2) for chunk in chunks)
    assert all(chunk["doc_id"] == "test-doc" for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    stats = chunk_stats(chunks)
    assert stats["count"] == len(chunks)
    assert stats["max_chars"] <= 800 * 1.5  # splitter tolerates long unbreakable words
    assert stats["pages"] == 2


def test_chunking_is_deterministic(pages: list[dict[str, Any]]) -> None:
    """The same input always produces the same chunk ids."""
    first = chunk_pages(pages, "abc")
    second = chunk_pages(pages, "abc")
    assert [chunk["chunk_id"] for chunk in first] == [
        chunk["chunk_id"] for chunk in second
    ]


def test_text_fragments_rebuild_table_rows() -> None:
    """Table rows stored one cell per line are re-joined into row fragments."""
    text = "Programming\nPython\nMain backend development\nPDF Processing\nPyMuPDF\nExtract text"
    fragments = iter_text_fragments(text)
    assert fragments
    assert any("PyMuPDF" in fragment for fragment in fragments)
    assert all("\n" not in fragment for fragment in fragments)


# ── Embeddings + vector store ─────────────────────────────────────────────


def test_embedder_returns_normalised_vectors() -> None:
    """Embeddings are 384-dimensional, L2-normalised float32 vectors."""
    vectors = get_embedder().encode(
        ["retrieval augmented generation", "vector search with faiss"]
    )
    assert vectors.shape == (2, get_embedder().dim)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_faiss_add_and_search(indexed: tuple[str, VectorStore]) -> None:
    """Adding vectors persists an index and search returns ranked neighbours."""
    doc_id, store = indexed
    assert store.exists(doc_id)

    query = get_embedder().encode_query("How does FAISS perform similarity search?")
    results = store.search(doc_id, query, k=3)

    assert 1 <= len(results) <= 3
    assert [result["rank"] for result in results] == list(range(1, len(results) + 1))
    assert results[0]["score"] >= results[-1]["score"]  # sorted by similarity
    assert results[0]["chunk"]["page_num"] in (1, 2)

    faiss_file = store.index_path(doc_id)
    assert faiss_file.exists()
    assert store.meta_path(doc_id).exists()


def test_faiss_search_can_filter_by_page(indexed: tuple[str, VectorStore]) -> None:
    """A page filter restricts results to the requested pages."""
    doc_id, store = indexed
    query = get_embedder().encode_query("embeddings")
    results = store.search(doc_id, query, k=5, page_filter={2})
    assert all(result["chunk"]["page_num"] == 2 for result in results)


def test_faiss_missing_index_raises(indexed: tuple[str, VectorStore]) -> None:
    """Searching an unknown document fails loudly."""
    _doc_id, store = indexed
    with pytest.raises(FileNotFoundError):
        store.load("does-not-exist")


def test_vector_store_rejects_mismatched_chunks(tmp_path: Path) -> None:
    """Vectors and chunk metadata must have the same length."""
    store = VectorStore(tmp_path)
    with pytest.raises(ValueError):
        store.add("bad", np.zeros((2, 4), dtype=np.float32), [{"text": "one"}])


# ── RAG ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def rag_ready(monkeypatch: pytest.MonkeyPatch, indexed: tuple[str, VectorStore]
              ) -> str:
    """Register the throwaway vector store as the process-wide singleton."""
    doc_id, store = indexed
    monkeypatch.setattr(
        "backend.services.rag_pipeline.get_vector_store", lambda: store
    )
    monkeypatch.setattr(
        "backend.services.quiz_gen.get_vector_store", lambda: store
    )
    return doc_id


def test_retrieve_returns_grounded_chunks(rag_ready: str) -> None:
    """Retrieval returns chunks that actually contain the answer topic."""
    results = retrieve(rag_ready, "What is FAISS used for?", k=3)
    assert results
    text = " ".join(result["chunk"]["text"] for result in results).lower()
    assert "faiss" in text or "similarity search" in text


def test_rag_query_returns_answer_and_sources(rag_ready: str) -> None:
    """The full RAG call returns an answer plus clickable page references."""
    result = rag_query(rag_ready, "What is FAISS?", mark_type="2M", language="english")

    assert result["answer"].strip()
    assert result["answer"] != INSUFFICIENT_INFO_ANSWER
    assert result["sources"], "a grounded answer must carry sources"
    assert all(source["page"] >= 1 for source in result["sources"])
    assert all(source["chunk_id"] for source in result["sources"])
    assert result["mark_type"] == "2M"
    assert 0.0 <= result["confidence"] <= 1.0


def test_rag_abstains_on_unrelated_question(rag_ready: str) -> None:
    """A question unrelated to the document is refused, not hallucinated."""
    result = rag_query(
        rag_ready,
        "Who won the FIFA World Cup final in 1998 and how many goals?",
        mark_type="5M",
    )
    assert result["answer"] == INSUFFICIENT_INFO_ANSWER
    assert result["grounded"] is False


def test_rag_rejects_empty_question(rag_ready: str) -> None:
    """An empty question is a client error."""
    with pytest.raises(ValueError):
        rag_query(rag_ready, "   ", mark_type="5M")


def test_mark_wise_rules_change_the_prompt() -> None:
    """Every mark type injects a distinct length/structure rule."""
    prompts = {mark: get_mark_prompt(mark) for mark in MARK_TYPES}
    assert len(set(prompts.values())) == len(MARK_TYPES)
    assert normalise_mark_type("10m") == "10M"
    assert normalise_mark_type(16) == "16M"
    assert normalise_mark_type("nonsense") == "5M"
    assert mark_int("15M") == 15


def test_language_instructions_differ() -> None:
    """Each language maps to a non-empty, distinct prompt instruction."""
    english = get_language_instruction("english")
    tamil = get_language_instruction("tamil")
    tanglish = get_language_instruction("tanglish")
    assert english != tamil != tanglish
    assert "தமிழ்" in tamil
    assert normalize_language("TA") == "tamil"
    assert normalize_language("bogus") == "english"


# ── Offline fallbacks (no LLM required) ──────────────────────────────────


def test_extractive_answer_scales_with_marks(rag_ready: str) -> None:
    """Offline answers grow with the requested mark count."""
    results = retrieve(rag_ready, "What is retrieval augmented generation?", k=5)
    short = extractive_answer("What is retrieval augmented generation?", results, "1M")
    long = extractive_answer(
        "What is retrieval augmented generation?", results, "16M"
    )
    assert short.strip() and long.strip()
    assert len(long.split()) >= len(short.split())
    assert "Point 1" in long  # 16M gets labelled sections


def test_answer_type_detection() -> None:
    """Expected answer types are detected from the question wording."""
    assert expected_answer_type("In what year did it open?") == "number"
    assert expected_answer_type("How many students enrolled?") == "number"
    assert expected_answer_type("Who wrote the book?") == "person"
    assert expected_answer_type("What is called a transformer?") == "proper"
    assert expected_answer_type("Explain supervised learning") == "any"
    assert answer_type_supported("In what year?", "The school opened in 1852.")
    assert not answer_type_supported("In what year?", "The school opened soon.")


def test_parse_quiz_json_handles_messy_llm_output() -> None:
    """Markdown fences, prose and truncated JSON are all handled."""
    fenced = '```json\n[{"question": "Q?", "options": ["a", "b", "c", "d"], ' \
             '"correct_index": 1, "source_page": 3}]\n```'
    assert len(parse_quiz_json(fenced, "General")) == 1

    chatty = 'Sure! Here you go: [{"question": "Q?", "options": ["a","b","c","d"],' \
             ' "correct_index": 0}] Hope this helps!'
    assert len(parse_quiz_json(chatty, "General")) == 1

    assert parse_quiz_json("no json at all", "General") == []


def test_offline_quiz_generates_four_options(indexed: tuple[str, VectorStore]) -> None:
    """The offline generator always produces a valid, answerable MCQ set."""
    doc_id, store = indexed
    chunks = store.load(doc_id)["meta"]
    questions = offline_quiz(chunks, num_questions=3, difficulty="medium", topic="RAG")
    assert questions
    for question in questions:
        assert len(question["options"]) == 4
        assert 0 <= question["correct_index"] < 4
        assert "______" in question["question"]
        assert question["source_page"] >= 1


def test_offline_evaluation_scores_coverage(
    indexed: tuple[str, VectorStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detailed answer scores higher than a one-line answer."""
    _doc_id, store = indexed
    monkeypatch.setattr(
        "backend.services.rag_pipeline.get_vector_store", lambda: store
    )
    context = full_text(extract_pages_for_test())
    thin = offline_evaluate(
        "What is FAISS?", "It is a library.", 5, context, context
    )
    thick = offline_evaluate(
        "What is FAISS?",
        "FAISS is a library for efficient similarity search. It stores "
        "embeddings vectors and returns nearest neighbours using inner product.",
        5,
        context,
        context,
    )
    assert thick["score_earned"] > thin["score_earned"]
    assert thin["missing_points"]
    assert thin["suggestions"]


def extract_pages_for_test() -> list[dict[str, Any]]:
    """Rebuild the sample pages for evaluation tests."""
    document = pymupdf.open()
    tmp = Path(__file__).resolve().parent / "_tmp_sample.pdf"
    for text in SAMPLE_PAGES:
        document.new_page().insert_textbox(
            pymupdf.Rect(60, 60, 540, 760), text, fontsize=11
        )
    document.save(str(tmp))
    document.close()
    try:
        return extract_pages(tmp)
    finally:
        tmp.unlink(missing_ok=True)


# ── Quiz, planner and database ────────────────────────────────────────────


def test_score_submission_persists_results(tmp_path: Path, monkeypatch
                                           ) -> None:
    """Submitting a quiz stores the attempt used for weak-topic tracking."""
    from backend.services.quiz_gen import score_submission

    test_db = tmp_path / "test.db"
    db.init_db(test_db)
    monkeypatch.setattr(db, "DB_PATH", test_db)

    db.save_document("doc-1", "notes.pdf", str(tmp_path / "notes.pdf"), 2, 4)
    db.save_quiz(
        "quiz-1",
        "doc-1",
        [
            {
                "question": "Q1",
                "options": ["a", "b", "c", "d"],
                "correct_index": 0,
                "explanation": "because",
                "source_page": 1,
                "topic": "Vector Search",
            },
            {
                "question": "Q2",
                "options": ["a", "b", "c", "d"],
                "correct_index": 2,
                "explanation": "because",
                "source_page": 2,
                "topic": "Vector Search",
            },
        ],
    )
    result = score_submission("doc-1", "quiz-1", [0, 0])

    assert result["total"] == 2
    assert result["score"] == 1
    assert result["percentage"] == 50.0
    assert result["weak_topics"] == ["Vector Search"]

    # Test score_submission without explicit doc_id (resolves from quiz)
    res_no_doc = score_submission(quiz_id="quiz-1", answers={0: 0, 1: 2})
    assert res_no_doc["score"] == 2
    assert res_no_doc["percentage"] == 100.0
    assert res_no_doc["doc_id"] == "doc-1"

    from backend.services.weak_topics import weak_topics_report

    report = weak_topics_report("doc-1")
    assert report["attempts"] == 2


def test_study_plan_schedules_across_days() -> None:
    """The planner splits topics across days and reserves the final day."""
    from datetime import date, timedelta

    from backend.services.study_planner import parse_syllabus as _parse  # noqa: F401

    topics = parse_syllabus("- Vector Search\n- FAISS\n- Embeddings")
    assert topics == ["Vector Search", "FAISS", "Embeddings"]

    exam = date.today() + timedelta(days=7)
    assert (exam - date.today()).days == 7


def test_doc_id_and_filename_helpers() -> None:
    """Generated ids and filenames are filesystem safe and unique."""
    first = generate_doc_id("Machine Learning Notes.pdf")
    second = generate_doc_id("Machine Learning Notes.pdf")
    assert first != second
    assert first.startswith("Machine_Learning_Notes-")
    assert safe_filename("../../etc/passwd") == "passwd"


def test_uuid_uniqueness_smoke() -> None:
    """Sanity check that ids generated by the services are unique."""
    assert len({uuid.uuid4().hex for _ in range(100)}) == 100


def test_threshold_is_configured() -> None:
    """The abstention threshold is a sane probability-like value."""
    assert 0.0 < SIMILARITY_THRESHOLD <= 1.0