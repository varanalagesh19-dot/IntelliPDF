"""IntelliPDF - Streamlit user interface.

Redesigned using the Think & Play Quiz Your Day by Bato design system.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

# Allow running as `streamlit run frontend/app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Streamlit Community Cloud keeps secrets in st.secrets, not os.environ, so
# mirror them before the backend config module reads the environment.
SECRET_KEYS = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_MODEL",
    "GEMINI_MODEL",
    "OLLAMA_URL",
    "OLLAMA_MODEL",
    "DEFAULT_LLM",
    "EMBEDDING_BACKEND",
    "EMBEDDING_MODEL",
    "SIMILARITY_THRESHOLD",
    "TOP_K",
    "HYBRID_DENSE_WEIGHT",
    "HYBRID_LEXICAL_WEIGHT",
    "MAX_UPLOAD_MB",
)
try:
    for _key in SECRET_KEYS:
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # noqa: BLE001 - no secrets configured is normal locally
    pass

from backend.services.mark_wise import MARK_TYPES, marks_distribution  # noqa: E402
from backend.utils.language import LANGUAGE_LABELS  # noqa: E402
from frontend.api_client import get_transport  # noqa: E402

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="IntelliPDF",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Google Fonts Import ──────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600;700&family=Walter+Turncoat&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── CSS Design Tokens & Styling ──────────────────────────────────────────
st.markdown(
    """<style>
:root {
  --color-white: #FFFDF9;
  --color-dark: #2D2D2D;
  --color-dark-blue: #0C5D8C;
  --color-blue: #0477BD;
  --color-green: #389975;
  --color-red: #E5644D;
  --color-orange: #F7914E;
  --color-yellow: #FFC500;
  --color-pink: #FC94AA;
  --color-border: #C4C1BC;
  --font-primary: 'Baloo 2', cursive, sans-serif;
  --font-secondary: 'Plus Jakarta Sans', sans-serif;
  --font-accent: 'Walter Turncoat', cursive;
  --text-xs: 13px;
  --text-sm: 15px;
  --text-base: 16px;
  --text-lg: 18px;
  --text-xl: 20px;
  --text-2xl: 41px;
  --space-1: 7px;
  --space-2: 15px;
  --space-3: 18px;
  --space-4: 29px;
  --space-5: 35px;
  --radius-pill: 55.875px;
  --radius-card: 18px;
  --shadow-sm: 0px 2.288px 4.575px 0px rgba(0,0,0,0.1);
  --motion-fast: 0.2s ease;
  --motion-slow: 0.5s ease;
}

html, body, .stApp {
  background-color: var(--color-white) !important;
  font-family: var(--font-secondary) !important;
  color: var(--color-dark) !important;
  font-size: var(--text-base) !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-primary) !important;
  font-weight: 700 !important;
  color: var(--color-dark-blue) !important;
}

.page-title {
  font-family: var(--font-accent) !important;
  font-size: var(--text-2xl) !important;
  color: var(--color-blue) !important;
}

[data-testid="stSidebar"] {
  background-color: var(--color-white) !important;
  border-right: 1px solid var(--color-border) !important;
}

div.stButton > button {
  background-color: var(--color-blue) !important;
  color: var(--color-white) !important;
  border-radius: var(--radius-pill) !important;
  padding: var(--space-2) var(--space-5) !important;
  font-weight: 600 !important;
  border: 1px solid var(--color-blue) !important;
  transition: transform var(--motion-fast), background-color var(--motion-fast) !important;
}
div.stButton > button:hover {
  background-color: var(--color-dark-blue) !important;
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-sm) !important;
}
div.stButton > button:focus {
  outline: 3px solid var(--color-yellow) !important;
  outline-offset: 2px !important;
}

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea {
  background-color: var(--color-white) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-pill) !important;
  padding: 12px var(--space-3) !important;
  color: var(--color-dark) !important;
}
.stTextArea > div > div > textarea { border-radius: var(--radius-card) !important; }
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--color-blue) !important;
  box-shadow: 0 0 0 3px rgba(4,119,189,0.15) !important;
}

[data-testid="stFileUploader"] section {
  background-color: var(--color-white) !important;
  border: 2px dashed var(--color-border) !important;
  border-radius: var(--radius-card) !important;
  padding: var(--space-4) !important;
}

.stTabs [data-baseweb="tab"] {
  font-family: var(--font-primary) !important;
  font-weight: 600 !important;
  color: var(--color-dark) !important;
  font-size: var(--text-lg) !important;
  background-color: var(--color-white) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-pill) !important;
  padding: var(--space-1) var(--space-3) !important;
}
.stTabs [aria-selected="true"] {
  background-color: var(--color-blue) !important;
  color: var(--color-white) !important;
  border-color: var(--color-blue) !important;
}

[data-testid="stMetric"] {
  background-color: var(--color-white) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-card) !important;
  padding: var(--space-4) !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricValue"] div {
  font-family: var(--font-primary) !important;
  font-weight: 700 !important;
  color: var(--color-blue) !important;
  font-size: var(--text-2xl) !important;
}

.stAlert {
  background-color: var(--color-white) !important;
  border-radius: var(--radius-card) !important;
  border: 1px solid var(--color-border) !important;
  border-left: 5px solid var(--color-green) !important;
  color: var(--color-dark) !important;
}

.custom-card {
  background: var(--color-white) !important;
  border: 1px solid var(--color-border) !important;
  border-radius: var(--radius-card) !important;
  padding: var(--space-4) !important;
  box-shadow: var(--shadow-sm) !important;
  margin-bottom: var(--space-3) !important;
}
.custom-card-answer { border-left: 5px solid var(--color-green) !important; }
.custom-card-highlight { border-left: 5px solid var(--color-blue) !important; }
.custom-card-warning { border-left: 5px solid var(--color-yellow) !important; }
.custom-card-error { border-left: 5px solid var(--color-red) !important; }

.badge-pill {
  display: inline-flex; align-items: center;
  border-radius: var(--radius-pill);
  padding: 4px 15px;
  font-size: var(--text-xs);
  font-weight: 600;
  margin: 2px 4px 2px 0;
}
.badge-blue { background-color: var(--color-blue); color: var(--color-white); }
.badge-green { background-color: var(--color-green); color: var(--color-white); }
.badge-yellow { background-color: var(--color-yellow); color: var(--color-dark); }
.badge-orange { background-color: var(--color-orange); color: var(--color-white); }
.badge-red { background-color: var(--color-red); color: var(--color-white); }
.badge-pink { background-color: var(--color-pink); color: var(--color-white); }

.exam-timer {
  font-family: var(--font-primary) !important;
  font-size: var(--text-2xl) !important;
  font-weight: 700 !important;
  color: var(--color-red) !important;
  text-align: center;
}
</style>""",
    unsafe_allow_html=True,
)


# ── API Helpers ──────────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs: Any) -> Any:
    """Call the backend and return parsed JSON, raising ``RuntimeError``."""
    transport = get_transport()
    try:
        response = transport.request(method, path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        if transport.mode == "http":
            raise RuntimeError(
                f"Cannot reach the IntelliPDF API at {API_URL}. "
                "Start it with: uvicorn backend.main:app --port 8000"
            ) from exc
        raise RuntimeError(f"IntelliPDF request failed: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(f"API error {response.status_code}: {detail}")
    return response.json()


@st.cache_data(ttl=30)
def fetch_health() -> dict[str, Any] | None:
    """Return ``/health`` or ``None`` when the backend is unreachable."""
    try:
        transport = get_transport()
        if not transport.reachable():
            return None
        return transport.request("GET", "/health", timeout=10).json()
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=10)
def fetch_documents() -> list[dict[str, Any]]:
    """Return the list of uploaded documents."""
    try:
        return api("GET", "/documents")
    except RuntimeError:
        return []


def clear_caches() -> None:
    """Drop cached document/answer state after a mutation."""
    fetch_documents.clear()


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-title">📘 IntelliPDF</div>
          <div class="sidebar-tagline">Think &amp; play quiz your day</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    health = fetch_health()
    if health is None:
        st.markdown(
            '<div class="badge-pill badge-red">API offline</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Expected backend at `{API_URL}`.")
    else:
        services = health["services"]
        active_llm = services["llm"]["active"]
        st.markdown(
            f'<div class="badge-pill badge-green">API online</div> '
            f'<div class="badge-pill badge-blue">{active_llm}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Embeddings: `{services['embedder']['model'].split('/')[-1]}` · "
            f"Mode: `{get_transport().mode}`"
        )

    st.markdown("---")

    language = st.selectbox(
        "Answer language",
        list(LANGUAGE_LABELS.keys()),
        format_func=lambda key: LANGUAGE_LABELS[key],
        help="Select language for responses",
    )

    documents = fetch_documents()
    doc_ids = [doc["doc_id"] for doc in documents]
    doc_id = st.selectbox(
        "Active document",
        doc_ids or [None],
        format_func=lambda value: (
            next(
                (d["filename"] for d in documents if d["doc_id"] == value),
                "No document uploaded",
            )
            if value
            else "No document uploaded"
        ),
        help="Choose document to study from",
    )

    if doc_id:
        meta = next(d for d in documents if d["doc_id"] == doc_id)
        st.markdown(
            f'<div style="margin-top: 8px;">'
            f'<span class="badge-pill badge-blue">{meta["num_pages"]} pages</span>'
            f'<span class="badge-pill badge-yellow">{meta["num_chunks"]} chunks</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"ID: `{meta['doc_id'][:16]}...`")

    st.markdown("---")

    if st.button("Upload new PDF", use_container_width=True):
        st.session_state["selected_tab_idx"] = 0
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="helper-text">'
        "Identifies topics for focused study; does not predict the actual "
        "question paper."
        "</div>",
        unsafe_allow_html=True,
    )
    if health:
        st.caption(f"v{health.get('version', '1.0.0')} · SQuAD 2.0 evaluated")


# ── Main Area Header ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom: 24px;">
      <div class="page-title">Think &amp; Play Quiz Your Day</div>
      <div class="page-subtitle">AI-powered personalized learning and exam preparation system</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_names = [
    "Upload",
    "Ask",
    "Quiz",
    "Exam",
    "Study Plan",
    "Weak Topics",
    "Revision",
    "Evaluate",
]
upload_tab, ask_tab, quiz_tab, exam_tab, plan_tab, weak_tab, revision_tab, eval_tab = st.tabs(
    tab_names
)


def require_doc() -> bool:
    """Show a friendly hint and return ``False`` when no document is selected."""
    if not doc_id:
        st.markdown(
            """
            <div class="custom-card custom-card-warning">
              <h3>No document active</h3>
              <p>Please upload or select a PDF from the sidebar to start studying.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return False
    return True


def show_sources(sources: list[dict[str, Any]]) -> None:
    """Render page chips and snippet accordions."""
    if not sources:
        return
    st.markdown("#### Sources")
    chips_html = '<div style="margin-bottom: 12px;">'
    for source in sources:
        page = source.get("page", 0)
        chips_html += f'<span class="badge-pill badge-blue">Page {page}</span> '
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    for source in sources:
        page = source.get("page", 0)
        heading = source.get("heading") or f"Snippet from page {page}"
        with st.expander(f"Page {page} - {heading[:50]}", expanded=False):
            st.write(source.get("snippet", ""))
            if doc_id:
                st.markdown(
                    f"[Open PDF at page {page}]"
                    f"({API_URL}/documents/{doc_id}/file#page={page})"
                )


# ── Tab 1: Upload ────────────────────────────────────────────────────────
with upload_tab:
    st.markdown("<h3>Upload a subject PDF</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.8, 1.2], gap="large")

    with col1:
        st.markdown(
            '<div class="custom-card">',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Choose a PDF file (ML, NLP, DBMS, Networks, or any subject)",
            type=["pdf"],
            accept_multiple_files=False,
            help="Files up to 50 MB supported",
        )
        syllabus = st.text_area(
            "Optional syllabus or high-priority topics (one per line)",
            placeholder="- Retrieval Augmented Generation\n- Vector search with FAISS\n- Evaluation metrics",
            height=130,
            help="Helps focus study plans and quiz generation",
        )
        process_btn = st.button("Process PDF", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if process_btn:
            if uploaded is None:
                st.warning("Please choose a PDF file first.")
            else:
                with st.spinner("Extracting text → chunking → embedding → indexing..."):
                    try:
                        result = api(
                            "POST",
                            "/upload",
                            files={
                                "file": (
                                    uploaded.name,
                                    uploaded.getvalue(),
                                    "application/pdf",
                                )
                            },
                            data={"syllabus": syllabus},
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        clear_caches()
                        st.success(f"{result['message']}")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Pages", result["num_pages"])
                        m2.metric("Chunks", result["num_chunks"])
                        m3.metric("Doc ID", result["doc_id"][:10] + "...")

    with col2:
        st.markdown(
            """
            <div class="custom-card custom-card-highlight">
              <h3>How it works</h3>
              <ol style="padding-left: 20px; margin-bottom: 0;">
                <li><b>PyMuPDF</b> extracts clean text page by page</li>
                <li>Header and footer boilerplate are automatically removed</li>
                <li><b>LangChain</b> chunks text with balanced overlaps</li>
                <li><b>MiniLM</b> generates 384-dimensional vector embeddings</li>
                <li><b>FAISS</b> stores vectors for instant semantic search</li>
                <li><b>SQLite</b> tracks metadata, study history, and weak topics</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if health:
            services = health["services"]
            st.markdown(
                f"""
                <div class="custom-card">
                  <h3>Active configuration</h3>
                  <p style="margin: 0;">
                    <b>Embedding model:</b> {services['embedder']['model']}<br>
                    <b>Active LLM:</b> {services['llm']['active']}<br>
                    <b>Fallback chain:</b> {' → '.join(services['llm']['chain'])}<br>
                    <b>Top K retrieval:</b> {services['rag']['top_k']}<br>
                    <b>Abstention threshold:</b> {services['rag']['similarity_threshold']}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if documents:
        st.markdown("<h3>Uploaded documents</h3>", unsafe_allow_html=True)
        st.dataframe(
            [
                {
                    "Document ID": doc["doc_id"][:12] + "...",
                    "File name": doc["filename"],
                    "Pages": doc["num_pages"],
                    "Chunks": doc["num_chunks"],
                    "Uploaded": doc["uploaded_at"],
                }
                for doc in documents
            ],
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 2: Ask ───────────────────────────────────────────────────────────
with ask_tab:
    st.markdown("<h3>Ask anything from your PDF</h3>", unsafe_allow_html=True)
    if require_doc():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        question = st.text_area(
            "Question",
            placeholder="e.g. What is Retrieval Augmented Generation and why is it useful?",
            height=90,
            help="Type your question based on the document",
        )

        c_mark, c_lang, c_btn = st.columns([1.6, 1.0, 1.0], gap="medium")
        with c_mark:
            mark_type = st.radio(
                "Marks format",
                MARK_TYPES,
                horizontal=True,
                help="Answer length and structure adjust to match mark allocation",
            )
            st.caption(marks_distribution(mark_type))
        with c_lang:
            st.caption("Language")
            st.markdown(f"<b>{LANGUAGE_LABELS[language]}</b>", unsafe_allow_html=True)
        with c_btn:
            st.write("")
            ask_clicked = st.button("Get answer", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if ask_clicked:
            if not question.strip():
                st.warning("Please type a question first.")
            else:
                with st.spinner("Retrieving context and writing answer..."):
                    try:
                        answer = api(
                            "POST",
                            "/ask",
                            json={
                                "doc_id": doc_id,
                                "question": question,
                                "mark_type": mark_type,
                                "language": language,
                            },
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["last_answer"] = answer

        answer = st.session_state.get("last_answer")
        if answer:
            confidence = answer.get("confidence", 0.0)
            m_cols = st.columns(4)
            m_cols[0].metric("Marks", answer["mark_type"])
            m_cols[1].metric("Confidence", f"{confidence:.2f}")
            m_cols[2].metric("Label", answer["confidence_label"].title())
            m_cols[3].metric("Time", f"{answer.get('elapsed_ms', 0)} ms")

            st.markdown(
                f"""
                <div class="custom-card custom-card-answer">
                  <h3>Answer ({answer['mark_type']} · {LANGUAGE_LABELS[answer['language']]})</h3>
                  <div style="font-size: 16px; line-height: 1.7; color: #2D2D2D; margin-bottom: 15px;">
                    {answer['answer']}
                  </div>
                  <hr style="border: none; border-top: 1px solid #C4C1BC; margin: 15px 0;">
                  <div style="font-size: 13px; color: #2D2D2D; opacity: 0.85;">
                    Generated by <span class="badge-pill badge-blue">{answer['provider']}</span> 
                    model: <code>{answer.get('model', '-')}</code>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            show_sources(answer.get("sources", []))


# ── Tab 3: Quiz ──────────────────────────────────────────────────────────
with quiz_tab:
    st.markdown("<h3>Auto-generated MCQ quiz</h3>", unsafe_allow_html=True)
    if require_doc():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        q1, q2, q3 = st.columns([1.5, 1, 1], gap="medium")
        with q1:
            quiz_topic = st.text_input(
                "Topic (optional)",
                placeholder="e.g. Vector search, Embeddings",
                help="Leave empty to cover the whole document",
            )
        with q2:
            num_questions = st.slider("Questions count", 1, 20, 5)
        with q3:
            difficulty = st.selectbox(
                "Difficulty level", ["easy", "medium", "hard"], index=1
            )

        gen_quiz_btn = st.button("Generate quiz", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if gen_quiz_btn:
            with st.spinner("Generating questions from your document..."):
                try:
                    quiz = api(
                        "POST",
                        "/quiz",
                        json={
                            "doc_id": doc_id,
                            "num_questions": num_questions,
                            "difficulty": difficulty,
                            "topic": quiz_topic,
                        },
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["quiz"] = quiz
                    st.session_state["quiz_answers"] = {}
                    st.session_state.pop("quiz_result", None)

        quiz = st.session_state.get("quiz")
        if quiz:
            st.markdown(
                f"""
                <div class="custom-card custom-card-highlight">
                  <h3>Quiz ready · {len(quiz['questions'])} questions</h3>
                  <p style="margin: 0;">
                    Difficulty: <span class="badge-pill badge-yellow">{quiz['difficulty']}</span> 
                    Provider: <span class="badge-pill badge-blue">{quiz['provider']}</span>
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            quiz_answers: dict[int, int] = st.session_state.setdefault(
                "quiz_answers", {}
            )
            for idx, item in enumerate(quiz["questions"]):
                st.markdown(f'<div class="custom-card">', unsafe_allow_html=True)
                st.markdown(f"**Q{idx + 1}. {item['question']}**")
                st.markdown(
                    f'<div style="margin-bottom: 12px;">'
                    f'<span class="badge-pill badge-blue">Page {item["source_page"]}</span>'
                    f'<span class="badge-pill badge-yellow">{item["topic"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                quiz_answers[idx] = st.radio(
                    f"Select option for Q{idx + 1}",
                    range(len(item["options"])),
                    format_func=lambda opt, it=item: f"{'ABCD'[opt]}. {it['options'][opt]}",
                    key=f"quiz_opt_{quiz['quiz_id']}_{idx}",
                    label_visibility="collapsed",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Submit quiz", use_container_width=True):
                ordered = [
                    quiz_answers.get(i, -1) for i in range(len(quiz["questions"]))
                ]
                try:
                    res = api(
                        "POST",
                        "/quiz/submit",
                        json={
                            "doc_id": doc_id,
                            "quiz_id": quiz["quiz_id"],
                            "answers": ordered,
                        },
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["quiz_result"] = res

            quiz_res = st.session_state.get("quiz_result")
            if quiz_res:
                score_color = (
                    "var(--color-green)"
                    if quiz_res["percentage"] >= 60
                    else "var(--color-red)"
                )
                st.markdown(
                    f"""
                    <div class="custom-card" style="border-left: 5px solid {score_color};">
                      <h3>Quiz completed</h3>
                      <div style="font-family: var(--font-primary); font-size: 35px; color: {score_color}; font-weight: 700;">
                        Score: {quiz_res['score']} / {quiz_res['total']} ({quiz_res['percentage']}%)
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(
                    min(1.0, quiz_res["percentage"] / 100),
                    text=f"{quiz_res['percentage']}% accuracy",
                )

                for idx, feedback in enumerate(quiz_res["per_question_feedback"]):
                    is_correct = feedback["is_correct"]
                    badge_cls = "badge-green" if is_correct else "badge-red"
                    badge_lbl = "Correct" if is_correct else "Incorrect"
                    with st.expander(
                        f"Q{idx + 1}: {badge_lbl} - {feedback['topic']} (page {feedback['source_page']})"
                    ):
                        st.markdown(f"**Question:** {feedback['question']}")
                        st.markdown(
                            f'<span class="badge-pill {badge_cls}">{badge_lbl}</span>',
                            unsafe_allow_html=True,
                        )
                        if feedback.get("explanation"):
                            st.info(feedback["explanation"])

                if quiz_res.get("weak_topics"):
                    weak_list = ", ".join(quiz_res["weak_topics"])
                    st.markdown(
                        f"""
                        <div class="custom-card custom-card-error">
                          <h3>Weak topics detected</h3>
                          <p>Consider revising: <b>{weak_list}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ── Tab 4: Exam Mode ─────────────────────────────────────────────────────
with exam_tab:
    st.markdown("<h3>Timed mock exam</h3>", unsafe_allow_html=True)
    if require_doc():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        e1, e2 = st.columns([1, 2], gap="large")
        with e1:
            duration = st.number_input(
                "Duration (minutes)",
                min_value=5,
                max_value=240,
                value=60,
                step=5,
                help="Exam timer limit",
            )
        with e2:
            st.caption("Marks distribution")
            dist = {
                mark: st.slider(
                    f"{mark} questions", 0, 10, default, key=f"exam_dist_{mark}"
                )
                for mark, default in (
                    ("1M", 5),
                    ("2M", 3),
                    ("5M", 2),
                    ("10M", 0),
                    ("15M", 0),
                    ("16M", 1),
                )
            }
        active_dist = {k: v for k, v in dist.items() if v > 0}
        start_exam_btn = st.button("Start exam", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if start_exam_btn:
            with st.spinner("Generating question paper and setting timer..."):
                try:
                    exam_data = api(
                        "POST",
                        "/mock-exam",
                        json={
                            "doc_id": doc_id,
                            "duration_minutes": int(duration),
                            "marks_distribution": active_dist,
                        },
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["exam"] = exam_data
                    st.session_state["exam_started"] = time.time()
                    st.session_state["exam_answers"] = {}
                    st.session_state.pop("exam_result", None)

        exam = st.session_state.get("exam")
        if exam:
            total_marks = exam["total_marks"]
            st.session_state["deadline"] = (
                st.session_state.get("exam_started", time.time())
                + exam["duration_minutes"] * 60
            )
            remaining = max(0, st.session_state["deadline"] - time.time())
            minutes, seconds = divmod(int(remaining), 60)

            st.markdown(
                f"""
                <div class="custom-card custom-card-highlight">
                  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                      <h3>Question paper ({total_marks} marks)</h3>
                      <p style="margin: 0;">{exam['instructions']}</p>
                    </div>
                    <div class="exam-timer">
                      ⏳ {minutes:02d}:{seconds:02d}
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            exam_answers: dict[str, str] = st.session_state.setdefault(
                "exam_answers", {}
            )
            for idx, q_item in enumerate(exam["questions"]):
                q_key = f"q{idx + 1}"
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown(f"**Question {idx + 1}. {q_item['question']}**")
                st.markdown(
                    f'<div style="margin-bottom: 8px;">'
                    f'<span class="badge-pill badge-blue">{q_item["mark_type"]} ({q_item["marks"]} marks)</span>'
                    f'<span class="badge-pill badge-yellow">Page {q_item["source_page"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                exam_answers[q_key] = st.text_area(
                    f"Answer for Q{idx + 1}",
                    value=exam_answers.get(q_key, ""),
                    height=110,
                    key=f"exam_inp_{exam['exam_id']}_{q_key}",
                    label_visibility="collapsed",
                )
                st.markdown("</div>", unsafe_allow_html=True)

            col_sub, col_reset = st.columns([2, 1], gap="medium")
            with col_sub:
                if st.button("Submit paper", use_container_width=True):
                    with st.spinner("Grading answers against source..."):
                        try:
                            graded = api(
                                "POST",
                                "/mock-exam/submit",
                                json={
                                    "exam_id": exam["exam_id"],
                                    "answers": exam_answers,
                                },
                            )
                        except RuntimeError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["exam_result"] = graded
            with col_reset:
                if st.button(
                    "New paper", use_container_width=True, kind="secondary"
                ):
                    for k in (
                        "exam",
                        "exam_result",
                        "exam_answers",
                        "deadline",
                        "exam_started",
                    ):
                        st.session_state.pop(k, None)
                    st.rerun()

            graded = st.session_state.get("exam_result")
            if graded:
                if graded.get("auto_submitted"):
                    st.warning("Time expired. Your paper was automatically submitted.")

                em1, em2, em3, em4 = st.columns(4)
                em1.metric(
                    "Total score", f"{graded['score']}/{graded['total_marks']}"
                )
                em2.metric("Percentage", f"{graded['percentage']}%")
                em3.metric("Time used", f"{graded['time_used_minutes']} min")
                em4.metric(
                    "Submission",
                    "Auto" if graded.get("auto_submitted") else "Manual",
                )

                st.progress(min(1.0, graded["percentage"] / 100))

                for r_item in graded.get("results", []):
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.markdown(
                        f"**{r_item['question_id']}: {r_item['question']}**"
                    )
                    st.markdown(
                        f'<div style="margin: 8px 0;">'
                        f'<span class="badge-pill badge-blue">{r_item["mark_type"]}</span>'
                        f'<span class="badge-pill badge-green">Score: {r_item["score_earned"]}/{r_item["score_out_of"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if r_item.get("missing_points"):
                        st.markdown("**Missing points:**")
                        st.markdown(
                            '<ul class="bullet-missing">'
                            + "".join(
                                f"<li>{pt}</li>"
                                for pt in r_item["missing_points"][:4]
                            )
                            + "</ul>",
                            unsafe_allow_html=True,
                        )
                    if r_item.get("suggestions"):
                        st.info("Tip: " + "; ".join(r_item["suggestions"][:2]))
                    st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 5: Study Plan ────────────────────────────────────────────────────
with plan_tab:
    st.markdown("<h3>AI study plan</h3>", unsafe_allow_html=True)
    if require_doc():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        sp1, sp2 = st.columns([1, 1], gap="medium")
        default_target = date.today() + timedelta(days=7)
        with sp1:
            exam_target_date = st.date_input("Exam date", value=default_target)
        with sp2:
            daily_hours = st.slider(
                "Daily study hours",
                min_value=0.5,
                max_value=12.0,
                value=3.0,
                step=0.5,
            )
        gen_plan_btn = st.button("Generate plan", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if gen_plan_btn:
            with st.spinner("Analyzing topics and balancing daily workload..."):
                try:
                    plan = api(
                        "POST",
                        "/study-plan",
                        json={
                            "doc_id": doc_id,
                            "exam_date": exam_target_date.isoformat(),
                            "daily_hours": daily_hours,
                        },
                    )
                except (RuntimeError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    st.session_state["plan"] = plan

        plan = st.session_state.get("plan")
        if plan:
            p_cols = st.columns(3)
            p_cols[0].metric("Days left", plan["days_left"])
            p_cols[1].metric("Total hours", f"{plan['total_hours']} h")
            p_cols[2].metric("Topics count", len(plan["topics"]))

            if plan.get("weak_topics"):
                weak_tags = "".join(
                    f'<span class="badge-pill badge-red">{wt}</span>'
                    for wt in plan["weak_topics"]
                )
                st.markdown(
                    f"""
                    <div class="custom-card custom-card-warning">
                      <h3>Priority focus (weighted 1.5x)</h3>
                      <div>{weak_tags}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Daily schedule")
            for day in plan.get("plan", []):
                st.markdown(
                    f"""
                    <div class="custom-card timeline-card">
                      <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge-pill badge-yellow">Day {day['day_num']} · {day['date']}</span>
                        <span style="font-family: var(--font-primary); font-size: 20px; font-weight: 700; color: var(--color-blue);">
                          {day['total_hours']} hours
                        </span>
                      </div>
                      <p style="font-weight: 600; color: var(--color-dark-blue); margin: 10px 0 6px 0;">
                        {day['focus']}
                      </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for topic in day.get("topics", []):
                    badge_cls = {
                        "high": "badge-red",
                        "medium": "badge-orange",
                        "low": "badge-blue",
                    }.get(topic.get("priority"), "badge-blue")
                    st.markdown(
                        f"- <span class='badge-pill {badge_cls}'>{topic.get('priority', 'normal').title()}</span> "
                        f"**{topic['name']}** — {topic['hours']} h _{topic.get('reason', '')}_",
                        unsafe_allow_html=True,
                    )

            st.caption(plan.get("disclaimer", ""))


# ── Tab 6: Weak Topics ───────────────────────────────────────────────────
with weak_tab:
    st.markdown("<h3>Weak topic analysis</h3>", unsafe_allow_html=True)
    if require_doc():
        ref_btn = st.button("Refresh analysis", use_container_width=True)

        if ref_btn or "weak_report" not in st.session_state:
            try:
                report = api("GET", f"/weak-topics/{doc_id}")
                st.session_state["weak_report"] = report
            except RuntimeError as exc:
                st.session_state["weak_report"] = None
                st.info("Take a quiz first to generate weak topic data.")

        report = st.session_state.get("weak_report")
        if report:
            w1, w2, w3 = st.columns(3)
            w1.metric("Quiz attempts", report["attempts"])
            w2.metric("Average score", f"{report['average_score'] * 100:.0f}%")
            w3.metric("Weak topics", len(report["weak_topics"]))

            if report.get("topics"):
                import pandas as pd
                import plotly.express as px

                frame = pd.DataFrame(report["topics"])
                frame = frame.sort_values("accuracy", ascending=True)

                chart = px.bar(
                    frame,
                    x="accuracy",
                    y="topic",
                    orientation="h",
                    color_discrete_sequence=["#E5644D"],
                    range_x=[0, 1],
                    title="Topic accuracy (lower values highlight weaker understanding)",
                )
                chart.update_layout(
                    plot_bgcolor="#FFFDF9",
                    paper_bgcolor="#FFFDF9",
                    height=380,
                    font=dict(
                        family="Plus Jakarta Sans, sans-serif",
                        color="#2D2D2D",
                        size=14,
                    ),
                    xaxis=dict(
                        gridcolor="#C4C1BC",
                        tickformat=".0%",
                        title="Accuracy",
                    ),
                    yaxis=dict(
                        gridcolor="#C4C1BC",
                        title="Topic",
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(chart, use_container_width=True)

                if report.get("weak_topics"):
                    weak_chips = " ".join(
                        f'<span class="badge-pill badge-red">{t}</span>'
                        for t in report["weak_topics"]
                    )
                    st.markdown(
                        f"""
                        <div class="custom-card custom-card-error">
                          <h3>Focus next on these topics</h3>
                          <div style="margin-top: 8px;">{weak_chips}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No quiz history recorded yet. Complete a quiz to analyze weak topics."
                )


# ── Tab 7: Revision ──────────────────────────────────────────────────────
with revision_tab:
    st.markdown("<h3>Revision flashcards</h3>", unsafe_allow_html=True)
    if require_doc():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1], gap="medium")
        with r1:
            rev_mode = st.selectbox(
                "Study mode",
                ["last_minute", "definitions", "key_concepts"],
                format_func=lambda val: {
                    "last_minute": "Last-minute mix",
                    "definitions": "Definitions",
                    "key_concepts": "Key concepts",
                }[val],
            )
        with r2:
            num_cards = st.slider("Number of cards", 3, 20, 6)

        gen_cards_btn = st.button("Generate flashcards", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if gen_cards_btn:
            with st.spinner("Distilling key takeaways into flashcards..."):
                try:
                    rev_data = api(
                        "POST",
                        "/revision",
                        json={
                            "doc_id": doc_id,
                            "mode": rev_mode,
                            "num_cards": num_cards,
                            "language": language,
                        },
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["revision_cards"] = rev_data

        rev_data = st.session_state.get("revision_cards")
        if rev_data:
            show_back = st.checkbox("Flip all cards (show definitions)", value=False)

            for idx, card in enumerate(rev_data.get("cards", [])):
                st.markdown(
                    f"""
                    <div class="flashcard-box">
                      <div class="flashcard-term">
                        {idx + 1}. {card['front']}
                      </div>
                      <div class="flashcard-definition" style="display: {'block' if show_back else 'none'}; margin-top: 10px;">
                        {card['back']}
                        <div style="margin-top: 8px;">
                          <span class="badge-pill badge-blue">Page {card.get('source_page', '-')}</span>
                        </div>
                      </div>
                      <div style="display: {'none' if show_back else 'block'}; font-size: 13px; color: #0477BD; margin-top: 6px;">
                        👉 Enable "Flip all cards" above to view definition
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button(
                "Shuffle new set", use_container_width=True, kind="secondary"
            ):
                st.session_state.pop("revision_cards", None)
                st.rerun()


# ── Tab 8: Evaluate ──────────────────────────────────────────────────────
with eval_tab:
    st.markdown("<h3>Answer evaluation</h3>", unsafe_allow_html=True)
    if require_doc():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        eval_q = st.text_input(
            "Exam question",
            placeholder="e.g. Explain FAISS indexing and similarity search",
            help="Enter the exact question from your test",
        )
        eval_ans = st.text_area(
            "Your answer",
            placeholder="Paste or write the answer you wrote in the exam...",
            height=140,
            help="Your written explanation",
        )
        eval_mark_type = st.selectbox("Marks allocation", MARK_TYPES, index=2)
        st.caption(marks_distribution(eval_mark_type))

        eval_btn = st.button("Evaluate answer", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if eval_btn:
            if not eval_q.strip() or not eval_ans.strip():
                st.warning("Please fill in both the question and your answer.")
            else:
                with st.spinner("Analyzing answer against reference text..."):
                    try:
                        eval_res = api(
                            "POST",
                            "/evaluate",
                            json={
                                "doc_id": doc_id,
                                "question": eval_q,
                                "student_answer": eval_ans,
                                "mark_type": eval_mark_type,
                            },
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["eval_result"] = eval_res

        eval_res = st.session_state.get("eval_result")
        if eval_res:
            pct = eval_res.get("percentage", 0)
            score_color = (
                "var(--color-green)" if pct >= 60 else "var(--color-red)"
            )

            st.markdown(
                f"""
                <div class="custom-card" style="border-left: 5px solid {score_color};">
                  <h3>Evaluation score</h3>
                  <div style="font-family: var(--font-primary); font-size: 41px; font-weight: 700; color: {score_color};">
                    {eval_res['score_earned']} / {eval_res['score_out_of']} ({pct}%)
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, pct / 100))

            st.markdown(
                f"""
                <div class="custom-card custom-card-highlight">
                  <h3>Model reference answer</h3>
                  <div style="font-size: 16px; line-height: 1.7; color: #2D2D2D;">
                    {eval_res['model_answer']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if eval_res.get("missing_points"):
                st.markdown(
                    """
                    <div class="custom-card custom-card-error">
                      <h3>Missing points</h3>
                      <ul class="bullet-missing">
                    """
                    + "".join(
                        f"<li>{pt}</li>" for pt in eval_res["missing_points"]
                    )
                    + """
                      </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if eval_res.get("incorrect_points"):
                st.markdown(
                    """
                    <div class="custom-card custom-card-warning">
                      <h3>Incorrect or inaccurate statements</h3>
                      <ul class="bullet-missing">
                    """
                    + "".join(
                        f"<li>{pt}</li>" for pt in eval_res["incorrect_points"]
                    )
                    + """
                      </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if eval_res.get("suggestions"):
                st.markdown(
                    """
                    <div class="custom-card custom-card-warning" style="border: 2px solid var(--color-yellow);">
                      <h3>Suggestions to improve</h3>
                      <ul>
                    """
                    + "".join(
                        f"<li>{sug}</li>" for sug in eval_res["suggestions"]
                    )
                    + """
                      </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            show_sources(eval_res.get("sources", []))


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top: 47px; padding-top: 18px; border-top: 1px solid var(--color-border); text-align: center;">
      <p class="helper-text">
        Foundations: NLP → Text Embeddings → Vector Search → RAG → LLMs · Evaluated on SQuAD 2.0<br>
        IntelliPDF identifies topics for focused study; it does not predict the actual question paper.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)