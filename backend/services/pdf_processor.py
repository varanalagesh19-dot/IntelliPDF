"""PDF text extraction with page numbers, using PyMuPDF (``fitz``).

Two clean-up passes matter for retrieval quality on real lecture PDFs:

1. **Repeated header/footer removal** - a line that appears on most pages
   (page numbers, course codes, "Unit 3 - continued") is boilerplate and only
   adds noise to the embedding space.
2. **Whitespace/hyphenation repair** - de-hyphenate words split across line
   breaks and collapse runs of blank lines so the chunker sees coherent text.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:  # PyMuPDF >= 1.24 exposes `pymupdf`; older wheels only expose `fitz`.
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover - legacy import path
    import fitz  # type: ignore

LOGGER = logging.getLogger(__name__)

#: A page must have at least this many characters to be considered useful.
MIN_PAGE_CHARS = 30
#: Fraction of pages a boilerplate line must appear on to be stripped.
BOILERPLATE_RATIO = 0.35
#: Boilerplate lines must be short - real headings are longer than this.
BOILERPLATE_MAX_LEN = 90
#: Pure numbers (page numbers) always look like this.
_PAGE_NUMBER_RE = re.compile(r"^[\s\-–—]*\d{1,4}[\s\-–—]*$")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_SPACES_RE = re.compile(r"[ \t]{2,}")


class PDFProcessingError(RuntimeError):
    """Raised when a PDF cannot be opened or contains no usable text."""


def _is_page_number(line: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(line.strip()))


def detect_boilerplate(pages: list[dict[str, Any]]) -> set[str]:
    """Return the set of normalised lines that are repeated across pages.

    Args:
        pages: List of ``{"page_num": int, "text": str}`` dicts.

    Returns:
        Normalised lines that look like running headers or footers.
    """
    counter: Counter[str] = Counter()
    for page in pages:
        for line in page["text"].splitlines():
            stripped = line.strip()
            if not stripped or _is_page_number(stripped):
                counter["\x00page-number"] += 1
                continue
            if len(stripped) <= BOILERPLATE_MAX_LEN:
                counter[re.sub(r"\s+", " ", stripped).lower()] += 1

    threshold = max(2, int(len(pages) * BOILERPLATE_RATIO))
    boilerplate = {line for line, count in counter.items() if count >= threshold}
    # A page-number "line" is always boilerplate when the PDF has >1 page.
    if len(pages) > 1 and "\x00page-number" in counter:
        boilerplate.add("\x00page-number")
    if boilerplate:
        LOGGER.info("Detected %d repeated header/footer lines", len(boilerplate))
    return boilerplate


def _clean_page_text(text: str, boilerplate: set[str]) -> str:
    """Remove boilerplate lines and normalise whitespace for one page."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _is_page_number(stripped):
            continue
        if re.sub(r"\s+", " ", stripped).lower() in boilerplate:
            continue
        kept.append(stripped)

    cleaned = "\n".join(kept)
    cleaned = _HYPHEN_BREAK_RE.sub(r"\1\2", cleaned)  # de-hyphenate line breaks
    cleaned = _SPACES_RE.sub(" ", cleaned)
    cleaned = _TRAILING_WS_RE.sub("", cleaned)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def extract_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract cleaned text from a PDF, one entry per page.

    Args:
        pdf_path: Path to the PDF file on disk.

    Returns:
        ``[{"page_num": 1, "text": "..."}, ...]`` for every page that has
        usable text (empty/image-only pages are skipped).

    Raises:
        FileNotFoundError: The file does not exist.
        PDFProcessingError: The file is not a PDF, is encrypted, or has no
            extractable text (most likely a scanned document).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise PDFProcessingError(f"Not a PDF file: {path.name}")

    try:
        document = fitz.open(str(path))
    except Exception as exc:  # pragma: no cover - PyMuPDF raises RuntimeError
        raise PDFProcessingError(f"Could not open PDF: {exc}") from exc

    try:
        if getattr(document, "is_encrypted", False) and not document.authenticate(""):
            raise PDFProcessingError("PDF is password protected.")

        raw_pages: list[dict[str, Any]] = []
        for index in range(document.page_count):
            page = document.load_page(index)
            raw_pages.append(
                {"page_num": index + 1, "text": page.get_text("text") or ""}
            )
    finally:
        document.close()

    boilerplate = detect_boilerplate(raw_pages)
    pages: list[dict[str, Any]] = []
    for page in raw_pages:
        cleaned = _clean_page_text(page["text"], boilerplate)
        if len(cleaned) >= MIN_PAGE_CHARS:
            pages.append({"page_num": page["page_num"], "text": cleaned})

    if not pages:
        raise PDFProcessingError(
            "No selectable text found - this looks like a scanned PDF. "
            "Run it through OCR first."
        )

    LOGGER.info("Extracted %d text pages from %s", len(pages), path.name)
    return pages


def extract_metadata(pdf_path: str | Path) -> dict[str, Any]:
    """Return simple PDF metadata (page count, title, author)."""
    path = Path(pdf_path)
    with fitz.open(str(path)) as document:
        raw = document.metadata or {}
        return {
            "page_count": document.page_count,
            "title": (raw.get("title") or "").strip(),
            "author": (raw.get("author") or "").strip(),
            "subject": (raw.get("subject") or "").strip(),
        }


def full_text(pages: list[dict[str, Any]]) -> str:
    """Concatenate pages into one string with ``[Page X]`` markers."""
    return "\n\n".join(
        f"[Page {page['page_num']}]\n{page['text']}" for page in pages
    )