"""Unit tests for PDF text extraction."""

import tempfile
from pathlib import Path

import pytest
import pymupdf

from app.rag.ingestion.pdf_processor import PDFProcessor, PDFProcessingError


def create_test_pdf(pages: list[str]) -> Path:
    """Create a temporary PDF with given page texts for testing."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


class TestPDFProcessor:
    """Tests for PDFProcessor."""

    def test_extract_text_single_page(self):
        """Test extracting text from a single-page PDF."""
        pdf_path = create_test_pdf(["Hello world, this is page 1."])
        try:
            processor = PDFProcessor()
            texts = processor.extract_text(pdf_path)
            assert len(texts) == 1
            assert "Hello world" in texts[0]
            assert "page 1" in texts[0]
        finally:
            pdf_path.unlink()

    def test_extract_text_multiple_pages(self):
        """Test extracting text from multi-page PDF."""
        pages = [
            "Page 1 content with some text.",
            "Page 2 has different content.",
            "Page 3 is the last page.",
        ]
        pdf_path = create_test_pdf(pages)
        try:
            processor = PDFProcessor()
            texts = processor.extract_text(pdf_path)
            assert len(texts) == 3
            for i, text in enumerate(texts):
                assert f"Page {i + 1}" in text
        finally:
            pdf_path.unlink()

    def test_extract_text_preserves_page_order(self):
        """Test that page order is preserved."""
        pages = ["First page", "Second page", "Third page"]
        pdf_path = create_test_pdf(pages)
        try:
            processor = PDFProcessor()
            texts = processor.extract_text(pdf_path)
            assert texts[0] == "First page"
            assert texts[1] == "Second page"
            assert texts[2] == "Third page"
        finally:
            pdf_path.unlink()

    def test_extract_text_handles_empty_page(self):
        """Test that empty pages return empty string (preserves index)."""
        pages = ["Page 1 content", "", "Page 3 content"]
        pdf_path = create_test_pdf(pages)
        try:
            processor = PDFProcessor()
            texts = processor.extract_text(pdf_path)
            assert len(texts) == 3
            assert texts[0] == "Page 1 content"
            assert texts[1] == ""
            assert texts[2] == "Page 3 content"
        finally:
            pdf_path.unlink()

    def test_extract_text_nonexistent_file(self):
        """Test error on nonexistent file."""
        processor = PDFProcessor()
        with pytest.raises(PDFProcessingError, match="not found"):
            processor.extract_text("/nonexistent/path.pdf")

    def test_extract_document_returns_document(self):
        """Test extract_document returns full Document with metadata."""
        pages = ["Test content for document"]
        pdf_path = create_test_pdf(pages)
        try:
            processor = PDFProcessor()
            doc = processor.extract_document(pdf_path)

            assert doc.metadata.document_id.startswith("doc_")
            assert doc.metadata.filename == pdf_path.name
            assert doc.metadata.file_hash
            assert doc.metadata.page_count == 1
            assert doc.metadata.file_size_bytes > 0
            assert len(doc.pages) == 1
            assert doc.pages[0] == "Test content for document"
        finally:
            pdf_path.unlink()

    def test_extract_document_file_hash_consistency(self):
        """Test that same file produces same hash."""
        pages = ["Consistent content"]
        pdf_path = create_test_pdf(pages)
        try:
            processor = PDFProcessor()
            doc1 = processor.extract_document(pdf_path)
            doc2 = processor.extract_document(pdf_path)
            assert doc1.metadata.file_hash == doc2.metadata.file_hash
        finally:
            pdf_path.unlink()

    def test_get_page_count(self):
        """Test getting page count without full extraction."""
        pages = ["Page 1", "Page 2", "Page 3"]
        pdf_path = create_test_pdf(pages)
        try:
            processor = PDFProcessor()
            count = processor.get_page_count(pdf_path)
            assert count == 3
        finally:
            pdf_path.unlink()

    def test_max_pages_limit(self):
        """Test max_pages parameter limits extraction."""
        pages = ["Page 1", "Page 2", "Page 3", "Page 4", "Page 5"]
        pdf_path = create_test_pdf(pages)
        try:
            processor = PDFProcessor(max_pages=3)
            texts = processor.extract_text(pdf_path)
            assert len(texts) == 3
        finally:
            pdf_path.unlink()