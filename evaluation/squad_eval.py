"""SQuAD 2.0 evaluation of the IntelliPDF RAG pipeline.

Each SQuAD paragraph is treated as a mini-document: it is chunked, embedded
and indexed exactly like an uploaded PDF, then every question is answered
through the production :func:`rag_pipeline.rag_query`.  Metrics follow the
official SQuAD script (Rajpurkar, Jia & Liang, 2018):

* **Exact Match / F1** on the answerable questions
* **Answerable accuracy** - fraction of answerable questions answered
* **Unanswerable accuracy** - fraction of unanswerable questions where the
  pipeline correctly abstains ("Insufficient information ...")

Usage::

    python evaluation/squad_eval.py --num-samples 100 --k 5
    python evaluation/squad_eval.py --num-samples 50 --llm          # via the API

Results are written to ``evaluation/squad_results.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import INSUFFICIENT_INFO_ANSWER, SIMILARITY_THRESHOLD  # noqa: E402
from backend.services.chunker import chunk_pages  # noqa: E402
from backend.services.embedder import get_embedder  # noqa: E402
from backend.services.mark_wise import MARK_TYPES  # noqa: E402
from backend.services.rag_pipeline import rag_query  # noqa: E402
from backend.services.vector_store import VectorStore  # noqa: E402

LOGGER = logging.getLogger("squad_eval")

RESULTS_PATH = Path(__file__).resolve().parent / "squad_results.json"

#: Mark rule used per question - SQuAD questions are short-answer questions,
#: so a 2M answer is the closest match to the reference extract.
DEFAULT_MARKS = "2M"


# ── Official SQuAD metric helpers ─────────────────────────────────────────


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles and collapse whitespace."""
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Return 1.0 when the normalised prediction equals the gold answer."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Return token-level F1 between a prediction and the gold answer."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    """Return ``True`` when the pipeline declined to answer."""
    text = (answer or "").lower()
    return INSUFFICIENT_INFO_ANSWER.lower() in text or "insufficient information" in text


# ── Dataset handling ─────────────────────────────────────────────────────


def load_squad(num_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Load a deterministic slice of the SQuAD 2.0 dev set.

    Args:
        num_samples: Number of questions to evaluate.
        seed: Shuffle seed so runs are reproducible.

    Returns:
        A list of ``{"id", "question", "context", "answers", "is_answerable"}``.
    """
    from datasets import load_dataset

    LOGGER.info("Downloading the SQuAD 2.0 dev split ...")
    # NB: `rajpurkar/squad` is v1.1 (no unanswerable questions); v2 must be
    # requested explicitly.
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    dataset = dataset.shuffle(seed=seed)

    samples: list[dict[str, Any]] = []
    for item in dataset:
        answers = item["answers"]["text"]
        samples.append(
            {
                "id": item["id"],
                "title": item.get("title", ""),
                "question": item["question"],
                "context": item["context"],
                "answers": answers,
                "is_answerable": bool(answers),
            }
        )
        if len(samples) >= num_samples:
            break
    return samples


# ── Indexing a context as a mini-document ─────────────────────────────────


def index_context(context: str, store: VectorStore) -> tuple[str, dict[str, Any]]:
    """Chunk, embed and index one SQuAD paragraph; return ``(doc_id, meta)``."""
    doc_id = f"squad_{uuid.uuid4().hex[:10]}"
    pages = [{"page_num": 1, "text": context}]
    chunks = chunk_pages(pages, doc_id)
    embeddings = get_embedder().encode([chunk["text"] for chunk in chunks])
    store.add(doc_id, embeddings, chunks)
    return doc_id, {"pages": 1, "chunks": len(chunks)}


# ── Evaluation ────────────────────────────────────────────────────────────


def evaluate(
    samples: list[dict[str, Any]],
    store: VectorStore,
    mark_type: str = DEFAULT_MARKS,
    language: str = "english",
    keep_examples: int = 5,
) -> dict[str, Any]:
    """Run the full pipeline over ``samples`` and return the metric report."""
    totals = Counter()
    per_sample: list[dict[str, Any]] = []
    provider_used: Counter[str] = Counter()

    for index, sample in enumerate(samples, start=1):
        doc_id, meta = index_context(sample["context"], store)
        try:
            result = rag_query(
                doc_id=doc_id,
                query=sample["question"],
                mark_type=mark_type,
                language=language,
            )
        finally:
            store.delete(doc_id)  # keep disk usage flat

        answer = result["answer"]
        abstained = is_abstention(answer)
        provider_used[result["provider"]] += 1
        totals["questions"] += 1

        if sample["is_answerable"]:
            totals["answerable"] += 1
            if not abstained:
                em = max(exact_match_score(answer, gold) for gold in sample["answers"])
                f1 = max(f1_score(answer, gold) for gold in sample["answers"])
                totals["answerable_answered"] += 1
                totals["em"] += em
                totals["f1"] += f1
            else:
                totals["abstained_on_answerable"] += 1
                em = f1 = 0.0
        else:
            totals["unanswerable"] += 1
            if abstained:
                totals["unanswerable_correct"] += 1

        if index <= keep_examples:
            per_sample.append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "gold": sample["answers"][:1],
                    "prediction": answer[:220],
                    "is_answerable": sample["is_answerable"],
                    "abstained": abstained,
                    "confidence": result["confidence"],
                    "em": round(em, 4),
                    "f1": round(f1, 4),
                    "pages": meta["pages"],
                    "chunks": meta["chunks"],
                }
            )
        if index % 10 == 0:
            LOGGER.info("evaluated %d/%d questions", index, len(samples))

    answerable = totals["answerable"]
    unanswerable = totals["unanswerable"]
    return {
        "config": {
            "samples": len(samples),
            "mark_type": mark_type,
            "language": language,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "mark_types_available": list(MARK_TYPES),
        },
        "totals": {
            "questions": totals["questions"],
            "answerable": answerable,
            "unanswerable": unanswerable,
            "answerable_answered": totals["answerable_answered"],
            "abstained_on_answerable": totals["abstained_on_answerable"],
            "unanswerable_correct": totals["unanswerable_correct"],
        },
        "metrics": {
            "exact_match": round(totals["em"] / answerable, 4) if answerable else 0.0,
            "f1": round(totals["f1"] / answerable, 4) if answerable else 0.0,
            "answerable_accuracy": round(
                totals["answerable_answered"] / answerable, 4
            )
            if answerable
            else 0.0,
            "unanswerable_accuracy": round(
                totals["unanswerable_correct"] / unanswerable, 4
            )
            if unanswerable
            else 0.0,
        },
        "providers": dict(provider_used),
        "examples": per_sample,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the evaluation."""
    metrics = report["metrics"]
    totals = report["totals"]
    print("\n" + "=" * 66)
    print("SQuAD 2.0 EVALUATION — IntelliPDF RAG pipeline")
    print("=" * 66)
    print(f"Questions evaluated : {totals['questions']}")
    print(f"  answerable        : {totals['answerable']}")
    print(f"  unanswerable      : {totals['unanswerable']}")
    print("-" * 66)
    print(f"Exact Match (answerable) : {metrics['exact_match']:.3f}")
    print(f"F1           (answerable) : {metrics['f1']:.3f}")
    print(f"Answerable accuracy      : {metrics['answerable_accuracy']:.3f}")
    print(f"Unanswerable accuracy    : {metrics['unanswerable_accuracy']:.3f}")
    print("-" * 66)
    print(f"Providers used : {report['providers']}")
    print("-" * 66)
    for example in report["examples"]:
        kind = "ANSWERABLE" if example["is_answerable"] else "UNANSWERABLE"
        print(f"[{kind}] {example['question']}")
        print(f"    gold  : {example['gold']}")
        print(f"    pred  : {' '.join(example['prediction'].split())[:150]}")
        print(f"    EM={example['em']:.1f} F1={example['f1']:.2f} "
              f"conf={example['confidence']:.3f}")
    print("=" * 66)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate IntelliPDF on SQuAD 2.0")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Questions to evaluate (default: 50)")
    parser.add_argument("--mark-type", default=DEFAULT_MARKS, choices=list(MARK_TYPES))
    parser.add_argument("--language", default="english")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(RESULTS_PATH))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )
    started = time.perf_counter()
    samples = load_squad(args.num_samples, seed=args.seed)
    LOGGER.info("Loaded %d SQuAD 2.0 questions", len(samples))

    store = VectorStore()  # temp indexes are deleted right after each question
    report = evaluate(samples, store, mark_type=args.mark_type, language=args.language)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    report["dataset"] = "rajpurkar/squad_v2 (validation split)"

    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_report(report)
    print(f"\nSaved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())