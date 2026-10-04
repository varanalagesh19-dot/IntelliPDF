"""``POST /upload`` - ingest a PDF end to end.

Save -> extract page text (PyMuPDF) -> clean -> chunk (LangChain) -> embed
(MiniLM) -> store (FAISS) -> record metadata (SQLite).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.config import MAX_UPLOAD_MB, UPLOAD_PATH, ensure_directories
from backend.models.schemas import DocumentInfo, UploadResponse
from backend.services.chunker import chunk_pages
from backend.services.embedder import get_embedder
from backend.services.pdf_processor import PDFProcessingError, extract_pages
from backend.services.vector_store import get_vector_store
from backend.utils import db
from backend.utils.metadata import generate_doc_id, safe_filename

LOGGER = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


def _to_info(row: dict[str, object]) -> DocumentInfo:
    """Map a ``documents`` row (PK column ``id``) onto the API model."""
    return DocumentInfo(
        doc_id=str(row.get("id", "")),
        filename=str(row.get("filename", "")),
        num_pages=int(row.get("num_pages", 0) or 0),
        num_chunks=int(row.get("num_chunks", 0) or 0),
        uploaded_at=str(row.get("uploaded_at", "")),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(..., description="Subject PDF (max 50 MB)"),
    syllabus: str = Form("", description="Optional syllabus / important topics"),
) -> UploadResponse:
    """Upload and index a PDF.

    Args:
        file: The PDF to process.
        syllabus: Optional syllabus text used to seed important topics.

    Returns:
        Document id, page count and chunk count.

    Raises:
        HTTPException: 400 for a bad file, 413 when it is too large, 500 when
            extraction or indexing fails.
    """
    ensure_directories()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    doc_id = generate_doc_id(file.filename)
    stored_path = UPLOAD_PATH / f"{doc_id}.pdf"

    try:
        payload = await file.read()
    finally:
        await file.close()

    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(payload) / 1048576:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.",
        )

    stored_path.write_bytes(payload)
    LOGGER.info("Saved upload %s (%d KB)", stored_path.name, len(payload) // 1024)

    # -- extract ---------------------------------------------------------
    try:
        pages = extract_pages(stored_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PDFProcessingError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("PDF extraction failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {exc}") from exc

    num_pages = max(page["page_num"] for page in pages)

    # -- chunk + embed ---------------------------------------------------
    chunks = chunk_pages(pages, doc_id)
    if not chunks:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail="The PDF produced no usable text after cleaning."
        )

    try:
        embeddings = get_embedder().encode([chunk["text"] for chunk in chunks])
        get_vector_store().add(doc_id, embeddings, chunks)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Indexing failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    db.save_document(
        doc_id=doc_id,
        filename=safe_filename(file.filename),
        file_path=str(stored_path),
        num_pages=num_pages,
        num_chunks=len(chunks),
        syllabus=(syllabus or "").strip(),
    )

    LOGGER.info(
        "Indexed %s as %s (%d pages, %d chunks)",
        file.filename,
        doc_id,
        num_pages,
        len(chunks),
    )
    return UploadResponse(
        doc_id=doc_id,
        filename=safe_filename(file.filename),
        num_pages=num_pages,
        num_chunks=len(chunks),
        status="processed",
        syllabus_chars=len(syllabus or ""),
        message=f"Indexed {num_pages} pages into {len(chunks)} chunks.",
    )


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    """List every uploaded document (newest first)."""
    return [_to_info(row) for row in db.list_documents()]


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str) -> DocumentInfo:
    """Return one document's metadata."""
    row = db.get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    return _to_info(row)


@router.get("/documents/{doc_id}/file")
def get_pdf_file(doc_id: str) -> FileResponse:
    """Stream the original PDF so the UI can open it at a cited page."""
    row = db.get_document(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="The stored PDF is missing on disk.")
    return FileResponse(path, media_type="application/pdf", filename=row["filename"])


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str]:
    """Delete a document, its index and its history."""
    removed = db.delete_document(doc_id)
    get_vector_store().delete(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc_id}")
    return {"status": "deleted", "doc_id": doc_id}