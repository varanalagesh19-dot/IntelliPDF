"""SQLite persistence layer for IntelliPDF.

A tiny, dependency-free wrapper around :mod:`sqlite3` that owns the schema
for documents, quizzes, quiz results and study plans.  All functions are
thread-safe (a connection is opened per call) and tolerate concurrent
requests from FastAPI + Streamlit.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import DB_PATH, ensure_directories

LOGGER = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INITIALISED = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    num_pages    INTEGER NOT NULL DEFAULT 0,
    num_chunks   INTEGER NOT NULL DEFAULT 0,
    syllabus     TEXT DEFAULT '',
    uploaded_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quizzes (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    topic           TEXT DEFAULT '',
    difficulty      TEXT DEFAULT 'medium',
    questions_json  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quiz_results (
    id                  TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    quiz_id             TEXT NOT NULL,
    score               INTEGER NOT NULL,
    total               INTEGER NOT NULL,
    weak_topics_json    TEXT DEFAULT '[]',
    details_json        TEXT DEFAULT '[]',
    taken_at            TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS study_plans (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    plan_json    TEXT NOT NULL,
    exam_date    TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_quiz_results_doc ON quiz_results (doc_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_doc ON quizzes (doc_id);
CREATE INDEX IF NOT EXISTS idx_study_plans_doc ON study_plans (doc_id);
"""


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with row access by name and WAL enabled."""
    ensure_directories()
    target = Path(db_path or DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create all tables/indexes if they do not exist yet (idempotent)."""
    global _INITIALISED
    with _INIT_LOCK:
        conn = get_connection(db_path)
        try:
            conn.executescript(SCHEMA)
            conn.commit()
            _INITIALISED = True
            LOGGER.info("SQLite ready at %s", db_path or DB_PATH)
        finally:
            conn.close()


def _ensure_ready() -> None:
    if not _INITIALISED:
        init_db()


# ── Documents ────────────────────────────────────────────────────────────


def save_document(
    doc_id: str,
    filename: str,
    file_path: str,
    num_pages: int,
    num_chunks: int,
    syllabus: str = "",
) -> None:
    """Insert (or replace) the metadata row for an uploaded PDF."""
    _ensure_ready()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO documents
                (id, filename, file_path, num_pages, num_chunks, syllabus, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename=excluded.filename,
                file_path=excluded.file_path,
                num_pages=excluded.num_pages,
                num_chunks=excluded.num_chunks,
                syllabus=excluded.syllabus,
                uploaded_at=excluded.uploaded_at
            """,
            (doc_id, filename, str(file_path), num_pages, num_chunks, syllabus, utc_now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_document(doc_id: str) -> dict[str, Any] | None:
    """Return one document row as a dict, or ``None`` when unknown."""
    _ensure_ready()
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_documents(limit: int = 100) -> list[dict[str, Any]]:
    """Return all uploaded documents, newest first."""
    _ensure_ready()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(doc_id: str) -> bool:
    """Delete a document and its dependent rows. Returns True if removed."""
    _ensure_ready()
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Quizzes ──────────────────────────────────────────────────────────────


def save_quiz(
    quiz_id: str,
    doc_id: str,
    questions: list[dict[str, Any]],
    topic: str = "",
    difficulty: str = "medium",
) -> None:
    """Persist a generated quiz so it can be submitted later."""
    _ensure_ready()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO quizzes
                (id, doc_id, topic, difficulty, questions_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                quiz_id,
                doc_id,
                topic,
                difficulty,
                json.dumps(questions, ensure_ascii=False),
                utc_now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_quiz(quiz_id: str) -> dict[str, Any] | None:
    """Return a stored quiz with ``questions`` decoded from JSON."""
    _ensure_ready()
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        data["questions"] = json.loads(data.pop("questions_json") or "[]")
        return data
    finally:
        conn.close()


# ── Quiz results ─────────────────────────────────────────────────────────


def save_quiz_result(
    result_id: str,
    doc_id: str,
    quiz_id: str,
    score: int,
    total: int,
    weak_topics: list[str],
    details: list[dict[str, Any]] | None = None,
) -> None:
    """Persist a scored quiz attempt for weak-topic aggregation."""
    _ensure_ready()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO quiz_results
                (id, doc_id, quiz_id, score, total, weak_topics_json,
                 details_json, taken_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                doc_id,
                quiz_id,
                int(score),
                int(total),
                json.dumps(weak_topics, ensure_ascii=False),
                json.dumps(details or [], ensure_ascii=False),
                utc_now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_quiz_results(doc_id: str) -> list[dict[str, Any]]:
    """Return every attempt for a document, newest first, JSON decoded."""
    _ensure_ready()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM quiz_results WHERE doc_id = ? ORDER BY taken_at DESC",
            (doc_id,),
        ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["weak_topics"] = json.loads(item.pop("weak_topics_json") or "[]")
            item["details"] = json.loads(item.pop("details_json") or "[]")
            results.append(item)
        return results
    finally:
        conn.close()


# ── Study plans ──────────────────────────────────────────────────────────


def save_study_plan(
    plan_id: str, doc_id: str, plan: dict[str, Any], exam_date: str
) -> None:
    """Persist a generated study plan."""
    _ensure_ready()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO study_plans (id, doc_id, plan_json, exam_date, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (plan_id, doc_id, json.dumps(plan, ensure_ascii=False), exam_date, utc_now()),
        )
        conn.commit()
    finally:
        conn.close()


def latest_study_plan(doc_id: str) -> dict[str, Any] | None:
    """Return the most recent study plan for a document (decoded)."""
    _ensure_ready()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM study_plans WHERE doc_id = ? ORDER BY created_at DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["plan"] = json.loads(data.pop("plan_json") or "{}")
        return data
    finally:
        conn.close()