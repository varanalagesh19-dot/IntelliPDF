"""Text embeddings.

Primary path: ``sentence-transformers/all-MiniLM-L6-v2`` (Reimers &
Gurevych, 2019) producing 384-dimensional vectors, L2-normalised so that a
FAISS ``IndexFlatIP`` inner product equals cosine similarity.

Fallback path: a deterministic hashing embedder (hashed uni/bi-grams with
sub-linear term frequency).  It keeps IntelliPDF fully runnable with **zero**
downloads on an offline laptop - logins, uploads, quizzes and exams all still
work, just with lexical instead of semantic matching.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Final

import numpy as np

from backend.config import EMBED_BATCH_SIZE, EMBEDDING_BACKEND, EMBEDDING_MODEL

LOGGER = logging.getLogger(__name__)

EMBED_DIM: Final[int] = 384
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]+")


class HashingEmbedder:
    """Deterministic offline embedder used when sentence-transformers is absent.

    Hashes word unigrams and bigrams into ``dim`` buckets with sub-linear term
    weighting, then L2-normalises.  Not semantic, but stable and dependency
    free.
    """

    name = "hashing-fallback"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        """Encode ``texts`` into a ``(len(texts), dim)`` float32 array."""
        return np.vstack([self._encode_one(text) for text in texts])

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vector
        features: list[str] = tokens + [
            f"{a}_{b}" for a, b in zip(tokens, tokens[1:])
        ]
        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + np.log(count))
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


class SentenceTransformerEmbedder:
    """Thin wrapper around a ``SentenceTransformer`` model."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # local import

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers v5; keep both for older installs.
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(getter())

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        """Encode ``texts`` into normalised float32 embeddings."""
        vectors = self._model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class Embedder:
    """Lazily-initialised embedder shared by the whole application."""

    def __init__(self) -> None:
        self._impl: SentenceTransformerEmbedder | HashingEmbedder | None = None
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------
    @property
    def model_name(self) -> str:
        """Name of the active backend (``sentence-transformers`` or fallback)."""
        return self._impl.name if self._impl else "not-loaded"

    @property
    def dim(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._impl.dim if self._impl else EMBED_DIM

    def load(self) -> None:
        """Load the model once; fall back to hashing if anything fails.

        ``EMBEDDING_BACKEND=hashing`` skips torch entirely (important on 1 GB
        hosts); ``=sentence-transformers`` forces MiniLM; the default ``auto``
        tries MiniLM and degrades to hashing if it cannot be loaded.
        """
        if self._impl is not None:
            return
        with self._lock:
            if self._impl is not None:
                return
            if EMBEDDING_BACKEND == "hashing":
                LOGGER.info("EMBEDDING_BACKEND=hashing - using offline embeddings.")
                self._impl = HashingEmbedder()
                return
            try:
                LOGGER.info("Loading embedding model %s ...", EMBEDDING_MODEL)
                self._impl = SentenceTransformerEmbedder(EMBEDDING_MODEL)
                LOGGER.info(
                    "Embedding model ready (%s, dim=%d)",
                    self._impl.model_name,
                    self._impl.dim,
                )
            except Exception as exc:  # noqa: BLE001 - any failure -> fallback
                if EMBEDDING_BACKEND == "sentence-transformers":
                    raise
                LOGGER.warning(
                    "sentence-transformers unavailable (%s). "
                    "Falling back to offline hashing embeddings.",
                    exc.__class__.__name__,
                )
                self._impl = HashingEmbedder()

    # -- api ------------------------------------------------------------
    def encode(self, texts: list[str] | str) -> np.ndarray:
        """Embed one or many texts into an L2-normalised ``float32`` matrix."""
        self.load()
        assert self._impl is not None
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._impl.encode(list(texts)), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query string and return a ``(1, dim)`` matrix."""
        cleaned = (query or "").strip()
        if not cleaned:
            return np.zeros((1, self.dim), dtype=np.float32)
        return self.encode([cleaned])


#: Module-level singleton used by the routers.
embedder = Embedder()


def get_embedder() -> Embedder:
    """Return the process-wide embedder singleton."""
    return embedder