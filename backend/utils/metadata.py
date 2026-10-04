"""Document metadata helpers: doc-id generation, filenames and JSON codecs."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def generate_doc_id(filename: str) -> str:
    """Return a stable, human-friendly document id.

    The id is ``<sanitised-stem>-<8 hex chars>`` so it is readable in logs and
    unique across uploads of identically named files.

    Args:
        filename: Original uploaded filename.

    Returns:
        A filesystem-safe document identifier.
    """
    stem = Path(filename or "document").stem
    stem = _UNSAFE_FILENAME.sub("_", stem).strip("_") or "document"
    suffix = uuid.uuid4().hex[:8]
    return f"{stem[:48]}-{suffix}"


def safe_filename(filename: str) -> str:
    """Return a sanitised filename that is safe to write inside the upload dir."""
    name = Path(filename or "document.pdf").name
    name = _UNSAFE_FILENAME.sub("_", name)
    return name or "document.pdf"


def file_fingerprint(path: Path) -> str:
    """Return a short sha256 digest of a file's first 64 KB (fast, stable)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(65536))
    return digest.hexdigest()[:16]


def chunk_id_for(doc_id: str, page_num: int, index: int) -> str:
    """Return the deterministic chunk identifier used in source chips."""
    return f"{doc_id}::p{page_num}::c{index}"


def iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_dumps(payload: Any) -> str:
    """Serialise ``payload`` to JSON that survives non-ASCII text (Tamil)."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def json_loads(raw: str | None, default: Any = None) -> Any:
    """Parse JSON, returning ``default`` instead of raising on bad input."""
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate(text: str, limit: int = 320) -> str:
    """Shorten ``text`` for UI display, collapsing whitespace and adding an ellipsis."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"