"""Vector store built on FAISS ``IndexFlatIP``.

Because every vector is L2-normalised at embedding time, the inner product of
two vectors *is* their cosine similarity (Johnson, Douze & Jégou, 2019), so
``IndexFlatIP`` gives exact - not approximate - similarity search.

Artifacts per document:

* ``data/vector_db/{doc_id}.faiss``     - the FAISS index
* ``data/vector_db/{doc_id}_meta.pkl``  - pickled chunk metadata (parallel order)

If FAISS cannot be imported (e.g. no wheel for the platform) an exact NumPy
cosine search is used instead, so retrieval keeps working.
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import VECTOR_DB_PATH

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - import-time branch
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False
    LOGGER.warning("FAISS unavailable - using NumPy cosine search fallback.")


class VectorStore:
    """Create, persist and query per-document vector indexes."""

    def __init__(self, base_path: str | Path = VECTOR_DB_PATH) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    # -- paths ----------------------------------------------------------
    def index_path(self, doc_id: str) -> Path:
        """Return the ``.faiss`` path for ``doc_id``."""
        return self.base_path / f"{doc_id}.faiss"

    def meta_path(self, doc_id: str) -> Path:
        """Return the ``_meta.pkl`` path for ``doc_id``."""
        return self.base_path / f"{doc_id}_meta.pkl"

    def exists(self, doc_id: str) -> bool:
        """Return ``True`` when a persisted index exists for ``doc_id``."""
        if not self.index_path(doc_id).exists():
            return False
        if FAISS_AVAILABLE:
            return True
        return self.meta_path(doc_id).exists()

    # -- write ----------------------------------------------------------
    def add(
        self, doc_id: str, embeddings: np.ndarray, chunks: list[dict[str, Any]]
    ) -> int:
        """Build the index for ``doc_id`` and persist it to disk.

        Args:
            doc_id: Document identifier.
            embeddings: ``(n, dim)`` float32 matrix, ideally L2-normalised.
            chunks: Chunk metadata dicts, in the same order as ``embeddings``.

        Returns:
            The number of vectors stored.
        """
        if len(chunks) == 0:
            raise ValueError("Cannot build an index from zero chunks.")
        matrix = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding/chunk mismatch: {matrix.shape[0]} vectors vs "
                f"{len(chunks)} chunks."
            )
        matrix = self._normalize(matrix)

        with self._lock:
            if FAISS_AVAILABLE:
                dim = int(matrix.shape[1])
                index = faiss.IndexFlatIP(dim)
                index.add(matrix)
                faiss.write_index(index, str(self.index_path(doc_id)))
            else:
                self._write_matrix(doc_id, matrix)
            with open(self.meta_path(doc_id), "wb") as handle:
                pickle.dump(chunks, handle)

            self._cache[doc_id] = {"index": None, "matrix": matrix, "meta": chunks}
        LOGGER.info("Stored %d vectors for %s (faiss=%s)", len(chunks), doc_id, FAISS_AVAILABLE)
        return len(chunks)

    # -- read -----------------------------------------------------------
    def load(self, doc_id: str) -> dict[str, Any]:
        """Load (and cache) the index + metadata for ``doc_id``.

        Raises:
            FileNotFoundError: No index has been created for this document.
        """
        with self._lock:
            cached = self._cache.get(doc_id)
            if cached is not None:
                return cached

            if not self.exists(doc_id):
                raise FileNotFoundError(f"No vector index for document {doc_id}")

            meta: list[dict[str, Any]]
            with open(self.meta_path(doc_id), "rb") as handle:
                meta = pickle.load(handle)

            state: dict[str, Any] = {"index": None, "matrix": None, "meta": meta}
            if FAISS_AVAILABLE:
                state["index"] = faiss.read_index(str(self.index_path(doc_id)))
            else:
                state["matrix"] = self._read_matrix(doc_id)

            self._cache[doc_id] = state
            return state

    def search(
        self,
        doc_id: str,
        query_vec: np.ndarray,
        k: int = 5,
        page_filter: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the ``k`` most similar chunks for a query vector.

        Args:
            doc_id: Document identifier.
            query_vec: ``(dim,)`` or ``(1, dim)`` query embedding.
            k: Number of neighbours to return.
            page_filter: Optional set of page numbers to restrict the search to
                (used by topic-scoped features).

        Returns:
            ``[{"chunk", "score", "rank"}]`` sorted by descending similarity.
        """
        state = self.load(doc_id)
        meta: list[dict[str, Any]] = state["meta"]
        if not meta:
            return []

        query = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        query = self._normalize(query.reshape(1, -1))[0]

        if page_filter:
            candidates = [i for i, m in enumerate(meta) if m.get("page_num") in page_filter]
            if not candidates:
                return []
        else:
            candidates = list(range(len(meta)))

        if state["index"] is not None:
            index = state["index"]
            k_eff = min(len(meta), max(k, 1))
            scores, ids = index.search(query.reshape(1, -1), k_eff)
            ranked = list(zip(ids[0].tolist(), scores[0].tolist()))
        else:
            matrix: np.ndarray = state["matrix"]
            sub = matrix[candidates]
            scores = sub @ query
            order = np.argsort(-scores)[:k]
            ranked = [(candidates[int(i)], float(scores[int(i)])) for i in order]

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if idx < 0 or idx >= len(meta) or idx not in candidates:
                continue
            results.append({"chunk": meta[idx], "score": float(score), "rank": rank})
        return results[:k]

    # -- delete ---------------------------------------------------------
    def delete(self, doc_id: str) -> bool:
        """Remove the persisted index for ``doc_id``. Returns True if deleted."""
        with self._lock:
            self._cache.pop(doc_id, None)
            removed = False
            for path in (self.index_path(doc_id), self.meta_path(doc_id)):
                if path.exists():
                    path.unlink()
                    removed = True
            return removed

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    def _write_matrix(self, doc_id: str, matrix: np.ndarray) -> None:
        np.save(str(self.base_path / f"{doc_id}.npy"), matrix)

    def _read_matrix(self, doc_id: str) -> np.ndarray:
        path = self.base_path / f"{doc_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"No vector index for document {doc_id}")
        return np.load(path)


#: Module-level singleton used by the routers.
vector_store = VectorStore()


def get_vector_store() -> VectorStore:
    """Return the process-wide vector store singleton."""
    return vector_store