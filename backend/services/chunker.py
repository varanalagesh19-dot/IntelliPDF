"""Text chunking for retrieval.

Uses LangChain's ``RecursiveCharacterTextSplitter`` with the syllabus defaults
(800 chars / 150 chars overlap).  Page numbers are preserved: the document is
split *per page* so every chunk can be traced back to an exact page for the
"Source: Page X" chip in the UI.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE
from backend.utils.metadata import chunk_id_for

LOGGER = logging.getLogger(__name__)

SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

#: Lines that look like a section title: "3.1. Support Vector Machines",
#: "UNIT - IV: Result Properties", "Chapter 5", "Definition of Entropy".
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:unit|chapter|module|topic|section|part|lesson)\s*[-–.]?\s*\w+"
    r"|\d+(?:\.\d+){0,3}[\.\)]?\s+\S"
    r"|[A-Z][A-Za-z0-9 ,&/()\-]{4,60}:"
    r")\s*$"
)


def detect_heading(text: str) -> str | None:
    """Return the nearest heading line in ``text`` (``None`` when absent).

    The last heading-like line inside the *preceding* 400 characters is
    preferred, so a chunk inherits the section it belongs to.
    """
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[:6]):
        if 3 < len(line) <= 90 and _HEADING_RE.match(line):
            return line
    return None


def build_splitter(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
                   ) -> RecursiveCharacterTextSplitter:
    """Return the configured recursive character splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )


def chunk_pages(
    pages: list[dict[str, Any]],
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split extracted pages into retrieval chunks with metadata.

    Args:
        pages: Output of :func:`backend.services.pdf_processor.extract_pages`.
        doc_id: Document id used to build stable chunk ids.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        ``[{"doc_id", "chunk_id", "page_num", "heading", "text", "index"}]``.
        Chunks that contain only punctuation/whitespace are dropped.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks: list[dict[str, Any]] = []
    page_headings: dict[int, str] = {}

    for page in pages:
        page_num = int(page["page_num"])
        page_text = page["text"]
        if not page_text.strip():
            continue

        # Remember the most recent heading seen on this page so a chunk that
        # starts mid-page still knows its section.
        heading = detect_heading(page_text) or page_headings.get(page_num - 1)

        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if len(piece) < 20:  # noise: bullet glyphs, stray numbers
                continue
            local_heading = detect_heading(piece) or heading
            if local_heading:
                page_headings[page_num] = local_heading
                heading = local_heading
            index = len(chunks)
            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id_for(doc_id, page_num, index),
                    "page_num": page_num,
                    "heading": local_heading,
                    "text": piece,
                    "index": index,
                }
            )

    LOGGER.info("Chunked %d pages into %d chunks", len(pages), len(chunks))
    return chunks


def chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return simple size statistics for a chunk list (used by /health)."""
    if not chunks:
        return {"count": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0, "pages": 0}
    sizes = [len(chunk["text"]) for chunk in chunks]
    return {
        "count": len(sizes),
        "avg_chars": round(sum(sizes) / len(sizes), 1),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "pages": len({chunk["page_num"] for chunk in chunks}),
    }


def chunk_for_context(chunk: dict[str, Any], max_chars: int = 1200) -> str:
    """Render a chunk as a context block with a ``[Page X]`` marker."""
    heading = chunk.get("heading")
    header = f"[Page {chunk['page_num']}]"
    if heading:
        header += f" — {heading}"
    body = chunk["text"][:max_chars]
    return f"{header}\n{body}"


#: Minimum words in a reconstructed table row.
FRAGMENT_MIN_WORDS = 14


def iter_text_fragments(text: str) -> list[str]:
    """Split extracted text into reviewable fragments.

    Lecture PDFs are a mix of prose paragraphs and pipe/line tables whose rows
    are stored one cell per line.  Prose is returned line by line; table-like
    blocks (many short lines) are re-joined into ~:data:`FRAGMENT_MIN_WORDS`
    word rows so downstream sentence splitting has something to work with.

    Args:
        text: Cleaned page or chunk text.

    Returns:
        A list of non-empty single-line fragments.
    """
    fragments: list[str] = []
    for block in re.split(r"\n\s*\n", text or ""):
        lines = [" ".join(line.split()) for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        looks_like_table = len(lines) > 3 and max(
            len(line.split()) for line in lines
        ) <= 12
        if not looks_like_table:
            fragments.extend(lines)
            continue
        buffer: list[str] = []
        for line in lines:
            buffer.append(line)
            if len(" ".join(buffer).split()) >= FRAGMENT_MIN_WORDS:
                fragments.append(" ".join(buffer))
                buffer = []
        if buffer:
            fragments.append(" ".join(buffer))
    return fragments