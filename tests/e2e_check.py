"""End-to-end smoke test against a *running* IntelliPDF API.

Usage (from the project root, with the server already listening on :8000)::

    python tests/e2e_check.py sample_features.pdf

It walks the whole demo flow: upload -> ask (3 mark types, 3 languages) ->
quiz -> submit -> weak topics -> study plan -> revision -> evaluate ->
mock exam -> submit.  Exits non-zero on the first failure so it can be used
in CI or before a project demo.
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import requests

# Windows consoles default to cp1252, which cannot print the Tamil / emoji
# output the app produces.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8000"
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    """Print one check line and return whether it passed."""
    print(f"  [{PASS if condition else FAIL}] {label}" + (f" - {detail}" if detail else ""))
    return condition


def main(pdf_path: str) -> int:
    """Run the full flow; return a process exit code."""
    failures = 0

    print("\n== 1. health ==")
    health = requests.get(f"{BASE}/health", timeout=30).json()
    failures += not check("status ok", health["status"] == "ok")
    print(
        f"      embedder={health['services']['embedder']['model']} "
        f"llm_available={health['services']['llm']['any_available']}"
    )

    print("\n== 2. upload ==")
    with open(pdf_path, "rb") as handle:
        response = requests.post(
            f"{BASE}/upload",
            files={"file": (pdf_path.split("/")[-1], handle, "application/pdf")},
            data={"syllabus": "- Retrieval Augmented Generation\n- Vector Search\n- Evaluation"},
            timeout=300,
        )
    if response.status_code != 200:
        print(f"  [FAIL] upload -> {response.status_code}: {response.text[:300]}")
        return 1
    upload = response.json()
    doc_id = upload["doc_id"]
    failures += not check("doc_id returned", bool(doc_id), doc_id)
    failures += not check(
        "pages + chunks", upload["num_pages"] > 0 and upload["num_chunks"] > 0,
        f"{upload['num_pages']} pages / {upload['num_chunks']} chunks",
    )

    print("\n== 3. ask (mark-wise + language) ==")
    for mark in ("1M", "5M", "16M"):
        answer = requests.post(
            f"{BASE}/ask",
            json={
                "doc_id": doc_id,
                "question": "What is retrieval augmented generation and how does it work?",
                "mark_type": mark,
                "language": "english",
            },
            timeout=180,
        ).json()
        length = len(answer["answer"].split())
        failures += not check(f"{mark} answer + sources", bool(answer["answer"]) and bool(answer["sources"]),
                              f"{length} words, {len(answer['sources'])} sources, conf={answer['confidence']}")
        print(f"      {mark}: {answer['answer'][:110].replace(chr(10), ' ')}...")

    for language in ("tamil", "tanglish"):
        answer = requests.post(
            f"{BASE}/ask",
            json={"doc_id": doc_id, "question": "Define RAG", "mark_type": "2M",
                  "language": language},
            timeout=180,
        ).json()
        failures += not check(f"{language} answer", bool(answer["answer"]))
        print(f"      {language}: {answer['answer'][:110].replace(chr(10), ' ')}...")

    print("\n== 4. abstention on an off-topic question ==")
    off_topic = requests.post(
        f"{BASE}/ask",
        json={"doc_id": doc_id, "question": "Who won the 2018 FIFA World Cup?",
              "mark_type": "5M", "language": "english"},
        timeout=120,
    ).json()
    abstained = "insufficient information" in off_topic["answer"].lower()
    failures += not check("abstains when unanswerable", abstained, off_topic["answer"][:80])

    print("\n== 5. quiz ==")
    quiz = requests.post(
        f"{BASE}/quiz",
        json={"doc_id": doc_id, "num_questions": 5, "difficulty": "medium", "topic": "RAG"},
        timeout=600,
    ).json()
    quiz_id = quiz["quiz_id"]
    questions = quiz["questions"]
    failures += not check("5 MCQs generated", len(questions) == 5,
                          f"got {len(questions)} via {quiz['provider']}")
    failures += not check(
        "each question has 4 options + page",
        all(len(q["options"]) == 4 and q["source_page"] > 0 for q in questions),
    )
    print(f"      Q1: {questions[0]['question'][:100]}")

    answers = [q["correct_index"] if index % 2 else (q["correct_index"] + 1) % 4
               for index, q in enumerate(questions)]
    submission = requests.post(
        f"{BASE}/quiz/submit",
        json={"doc_id": doc_id, "quiz_id": quiz_id, "answers": answers},
        timeout=120,
    ).json()
    failures += not check(
        "score returned",
        submission["total"] == len(questions) and submission["total"] > 0,
        f"{submission['score']}/{submission['total']} = {submission['percentage']}%",
    )
    failures += not check(
        "per-question feedback",
        len(submission["per_question_feedback"]) == len(questions),
    )

    print("\n== 6. weak topics ==")
    weak = requests.get(f"{BASE}/weak-topics/{doc_id}", timeout=60).json()
    failures += not check("topic accuracy computed", len(weak["topics"]) > 0,
                          f"weak={weak['weak_topics']}")

    print("\n== 7. study plan ==")
    plan = requests.post(
        f"{BASE}/study-plan",
        json={"doc_id": doc_id,
              "exam_date": (date.today() + timedelta(days=6)).isoformat(),
              "daily_hours": 3},
        timeout=180,
    ).json()
    failures += not check("plan has days", len(plan["plan"]) >= 5,
                          f"{len(plan['plan'])} days, {plan['total_hours']} h")
    print(f"      Day 1: {plan['plan'][0]['focus'][:80]}")
    print(f"      Last : {plan['plan'][-1]['focus'][:80]}")

    print("\n== 8. revision ==")
    revision = requests.post(
        f"{BASE}/revision",
        json={"doc_id": doc_id, "mode": "definitions", "num_cards": 5, "language": "english"},
        timeout=600,
    ).json()
    failures += not check("cards generated", len(revision["cards"]) >= 3,
                          f"{len(revision['cards'])} cards")
    if revision["cards"]:
        print(f"      {revision['cards'][0]['front'][:70]} -> {revision['cards'][0]['back'][:70]}")

    print("\n== 9. evaluate ==")
    evaluation = requests.post(
        f"{BASE}/evaluate",
        json={"doc_id": doc_id,
              "question": "What is retrieval augmented generation?",
              "student_answer": "It is a technique.",
              "mark_type": "5M"},
        timeout=180,
    ).json()
    failures += not check("partial score + missing points",
                          evaluation["score_earned"] < evaluation["score_out_of"]
                          and len(evaluation["missing_points"]) > 0,
                          f"{evaluation['score_earned']}/{evaluation['score_out_of']}")

    print("\n== 10. mock exam ==")
    started = time.perf_counter()
    exam = requests.post(
        f"{BASE}/mock-exam",
        json={"doc_id": doc_id, "duration_minutes": 30,
              "marks_distribution": {"1M": 2, "5M": 1, "16M": 1}},
        timeout=900,
    ).json()
    build_seconds = time.perf_counter() - started
    failures += not check("paper built", len(exam["questions"]) == 4,
                          f"{exam['total_marks']} marks in {build_seconds:.1f}s")
    failures += not check("answer key present", len(exam["answer_key"]) == 4)
    print(f"      Q1 ({exam['questions'][0]['mark_type']}): {exam['questions'][0]['question'][:90]}")

    state = requests.get(f"{BASE}/mock-exam/{exam['exam_id']}", timeout=60).json()
    failures += not check("timer running", state["time_remaining_seconds"] > 0,
                          f"{state['time_remaining_seconds']:.0f}s left")

    answered = {
        f"q{index + 1}": "Retrieval augmented generation combines a retriever with a "
                         "generator so answers are grounded in the source document."
        for index in range(len(exam["questions"]))
    }
    graded = requests.post(
        f"{BASE}/mock-exam/submit",
        json={"exam_id": exam["exam_id"], "answers": answered},
        timeout=600,
    ).json()
    failures += not check("paper graded", graded["total_marks"] == exam["total_marks"],
                          f"{graded['score']}/{graded['total_marks']} "
                          f"in {graded['time_used_minutes']} min")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else str(failures) + ' CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_features.pdf"
    raise SystemExit(main(pdf))