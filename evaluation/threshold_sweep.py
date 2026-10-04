"""Calibrate the abstention threshold of the RAG pipeline.

Runs a small labelled probe set (questions that *are* answered by the uploaded
PDF vs. questions that are not) and reports, for every candidate threshold:
answerable recall, unanswerable precision and F1.  Use the printed table to
pick ``SIMILARITY_THRESHOLD`` in ``.env``.

Usage::

    python evaluation/threshold_sweep.py sample_rag.pdf
"""

from __future__ import annotations

import sys
from typing import Any

import requests

BASE = "http://127.0.0.1:8000"

ANSWERABLE = [
    "What tool is used for PDF text extraction?",
    "How are chunks converted into vectors?",
    "Which vector database is used?",
    "What is the role of LangChain?",
    "Which embedding model is used?",
    "How are answers generated from the retrieved context?",
    "What framework is used for the backend API?",
]

UNANSWERABLE = [
    "Who won the 2018 FIFA World Cup?",
    "Explain the life cycle of a butterfly in detail.",
    "What is the capital of France and why?",
    "Write a poem about the monsoon season.",
    "How do I bake a chocolate cake?",
    "Who invented the telephone and in which year?",
    "Explain the water cycle in three points.",
]


def upload(pdf_path: str) -> str:
    """Upload a PDF and return its ``doc_id``."""
    with open(pdf_path, "rb") as handle:
        response = requests.post(
            f"{BASE}/upload",
            files={"file": (pdf_path.split("/")[-1], handle, "application/pdf")},
            timeout=300,
        )
    response.raise_for_status()
    return response.json()["doc_id"]


def probe(doc_id: str, question: str) -> float:
    """Return the top combined retrieval score for a question."""
    response = requests.get(
        f"{BASE}/ask/preview/{doc_id}", params={"query": question}, timeout=120
    )
    response.raise_for_status()
    results = response.json()["results"]
    return results[0]["score"] if results else 0.0


def sweep(doc_id: str) -> list[dict[str, Any]]:
    """Score both probe sets and return per-threshold metrics."""
    answerable = [(question, probe(doc_id, question)) for question in ANSWERABLE]
    unanswerable = [(question, probe(doc_id, question)) for question in UNANSWERABLE]

    print("\n== answerable questions (higher score = better retrieval) ==")
    for question, score in answerable:
        print(f"   {score:.3f}  {question}")
    print("\n== unanswerable questions (should stay low) ==")
    for question, score in unanswerable:
        print(f"   {score:.3f}  {question}")

    rows: list[dict[str, Any]] = []
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        answered_ok = sum(1 for _, s in answerable if s >= threshold)
        false_positives = sum(1 for _, s in unanswerable if s >= threshold)
        recall = answered_ok / len(answerable)
        specificity = 1 - false_positives / len(unanswerable)
        f1 = (
            2 * recall * specificity / (recall + specificity)
            if (recall + specificity)
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "answerable_recall": round(recall, 3),
                "unanswerable_accuracy": round(specificity, 3),
                "f1": round(f1, 3),
            }
        )
    return rows


def main() -> int:
    """Run the sweep and print the threshold table."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    doc_id = upload(sys.argv[1])
    print(f"Uploaded {sys.argv[1]} as {doc_id}")
    rows = sweep(doc_id)
    print("\n== threshold sweep ==")
    print(f"{'threshold':>10} {'answerable':>12} {'unanswerable':>14} {'f1':>7}")
    for row in rows:
        print(
            f"{row['threshold']:>10.2f} {row['answerable_recall']:>12.2f} "
            f"{row['unanswerable_accuracy']:>14.2f} {row['f1']:>7.2f}"
        )
    best = max(rows, key=lambda row: row["f1"])
    print(f"\nBest F1 at threshold {best['threshold']:.2f} (F1 {best['f1']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())