"""PDF text extraction using PyMuPDF.

This module provides a clean interface for extracting text from PDF files
page by page, preserving page numbers and handling edge cases.
"""

from pathlib import Path

import pymupdf

from app.models.document import Document
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""
    pass


class PDFProcessor:
    """Extracts text from PDF files page by page.

    Uses PyMuPDF (fitz) for reliable text extraction. Handles:
    - Multi-page PDFs
    - Empty pages
    - Corrupted/unreadable pages (logs warning, continues)
    - Page number preservation
    """

    def __init__(self, max_pages: int | None = None):
        """Initialize processor.

        Args:
            max_pages: Optional limit on pages to process (for testing).
        """
        self.max_pages = max_pages

    def extract_text(self, pdf_path: str | Path) -> list[str]:
        """Extract text from each page of a PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of page texts, one per page. Empty string for failed/empty pages.

        Raises:
            PDFProcessingError: If PDF cannot be opened or is invalid.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFProcessingError(f"PDF file not found: {pdf_path}")

        try:
            doc = pymupdf.open(pdf_path)
        except Exception as e:
            raise PDFProcessingError(f"Failed to open PDF: {e}") from e

        if doc.page_count == 0:
            doc.close()
            raise PDFProcessingError("PDF has no pages")

        page_texts = []
        max_pages = self.max_pages or doc.page_count

        for page_num in range(min(max_pages, doc.page_count)):
            try:
                page = doc[page_num]
                text = page.get_text("text")  # Extract plain text
                page_texts.append(text.strip())
            except Exception as e:
                logger.warning(
                    "Failed to extract text from page",
                    extra={"page_number": page_num + 1, "error": str(e)},
                )
                page_texts.append("")  # Preserve page index alignment

        doc.close()
        logger.info(
            "PDF text extraction complete",
            extra={"pages_processed": len(page_texts), "path": str(pdf_path)},
        )
        return page_texts

    def extract_document(self, pdf_path: str | Path) -> Document:
        """Extract full document with metadata and page texts.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Document with metadata and page texts.

        Raises:
            PDFProcessingError: If PDF cannot be processed.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFProcessingError(f"PDF file not found: {pdf_path}")

        # Read file content (Document.create computes hash internally)
        content = pdf_path.read_bytes()

        # Extract page texts
        page_texts = self.extract_text(pdf_path)

        # Create document
        document = Document.create(
            filename=pdf_path.name,
            content=content,
            pages=page_texts,
        )

        logger.info(
            "Document extracted",
            extra={
                "document_id": document.metadata.document_id,
                "file_name": document.metadata.filename,
                "page_count": document.metadata.page_count,
                "file_hash": document.metadata.file_hash,
            },
        )
        return document

    def get_page_count(self, pdf_path: str | Path) -> int:
        """Get page count without full extraction."""
        pdf_path = Path(pdf_path)
        try:
            doc = pymupdf.open(pdf_path)
            count: int = doc.page_count  # type: ignore[attr-defined]
            doc.close()
            return count
        except Exception as e:
            raise PDFProcessingError(f"Failed to read PDF: {e}") from e
