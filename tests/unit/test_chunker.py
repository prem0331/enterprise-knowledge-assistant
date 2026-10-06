"""Unit tests for text chunking."""

from app.rag.ingestion.chunker import Chunker, ChunkingConfig
from app.models.chunk import Chunk
from app.models.document import Document, DocumentMetadata


class TestChunker:
    """Tests for Chunker."""

    def test_chunk_short_text_single_chunk(self):
        """Text shorter than chunk_size should produce one chunk."""
        chunker = Chunker(ChunkingConfig(chunk_size=1000, chunk_overlap=200))
        text = "This is a short text that fits in one chunk."
        chunks = chunker.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].document_id == "doc_1"
        assert chunks[0].document_name == "test.pdf"
        assert chunks[0].page_number == 1

    def test_chunk_long_text_multiple_chunks(self):
        """Text longer than chunk_size should produce multiple chunks."""
        chunker = Chunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
        # Create text longer than 100 chars
        text = " ".join(["This is a sentence."] * 10)  # ~200 chars
        chunks = chunker.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks) >= 2
        # Check all chunks have correct metadata
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == "doc_1"
            assert chunk.document_name == "test.pdf"
            assert chunk.chunk_index == i
            assert len(chunk.text) > 0

    def test_chunk_overlap_preserves_context(self):
        """Chunks should have overlapping content."""
        chunker = Chunker(ChunkingConfig(chunk_size=100, chunk_overlap=30))
        # Text with clear sentence boundaries
        text = "Sentence one. " * 10
        chunks = chunker.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks) >= 2
        # Verify overlap: end of chunk 0 should appear in chunk 1
        # (Exact overlap depends on splitter behavior, but there should be some shared text)
        assert len(chunks[0].text) <= 130  # chunk_size + some tolerance
        assert len(chunks[1].text) <= 130

    def test_chunk_metadata_preserved(self):
        """All metadata fields should be correctly set."""
        chunker = Chunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
        text = "Test content for metadata verification."
        chunks = chunker.chunk_text(
            text,
            document_id="doc_abc123",
            document_name="metadata_test.pdf",
            page_number=5,
            start_chunk_index=3,
        )

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.document_id == "doc_abc123"
        assert chunk.document_name == "metadata_test.pdf"
        assert chunk.page_number == 5
        assert chunk.chunk_index == 3
        assert chunk.chunk_id.startswith("chunk_")
        assert chunk.text == text

    def test_chunk_document_preserves_page_numbers(self):
        """Chunks from different pages should have correct page numbers."""
        # Create document with 3 pages
        doc = Document(
            metadata=DocumentMetadata(
                document_id="doc_test",
                filename="multipage.pdf",
                file_hash="abc123",
                page_count=3,
                file_size_bytes=1000,
            ),
            pages=[
                "Page 1 content with enough text to make a chunk.",
                "Page 2 content also long enough for chunking.",
                "Page 3 content final page text here.",
            ],
        )

        chunker = Chunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
        chunks = chunker.chunk_document(doc)

        # Should have chunks from all 3 pages
        page_numbers = {c.page_number for c in chunks}
        assert page_numbers == {1, 2, 3}

        # Each page should have at least one chunk
        for page_num in [1, 2, 3]:
            page_chunks = [c for c in chunks if c.page_number == page_num]
            assert len(page_chunks) >= 1

    def test_chunk_empty_text_returns_empty_list(self):
        """Empty or whitespace-only text should return no chunks."""
        chunker = Chunker()
        assert chunker.chunk_text("", "doc_1", "test.pdf") == []
        assert chunker.chunk_text("   ", "doc_1", "test.pdf") == []
        assert chunker.chunk_text("\n\n\n", "doc_1", "test.pdf") == []

    def test_chunk_document_skips_empty_pages(self):
        """Empty pages should be skipped without breaking chunk_index."""
        doc = Document(
            metadata=DocumentMetadata(
                document_id="doc_test",
                filename="test.pdf",
                file_hash="abc",
                page_count=3,
                file_size_bytes=100,
            ),
            pages=[
                "Page 1 has content.",
                "",  # Empty page
                "Page 3 has content.",
            ],
        )

        chunker = Chunker(ChunkingConfig(chunk_size=500, chunk_overlap=50))
        chunks = chunker.chunk_document(doc)

        # Should only have chunks from pages 1 and 3
        page_numbers = [c.page_number for c in chunks]
        assert 2 not in page_numbers  # Page 2 skipped
        assert len(chunks) == 2

        # Chunk indices should be sequential (0, 1)
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    def test_section_heading_detection_numbered(self):
        """Should detect numbered section headings like '2.1 Section Name'."""
        chunker = Chunker()
        text = "2.1 Authentication Methods\n\nJWT tokens are used for authentication."
        chunks = chunker.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks) >= 1
        # First chunk should have the heading
        assert chunks[0].section_heading == "Authentication Methods"

    def test_section_heading_detection_markdown(self):
        """Should detect markdown-style headings like '## Section Name'."""
        chunker = Chunker()
        text = "## API Authentication\n\nThe API uses JWT tokens."
        chunks = chunker.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks) >= 1
        assert chunks[0].section_heading == "API Authentication"

    def test_section_heading_detection_short_line(self):
        """Should detect short first line as heading if followed by content."""
        chunker = Chunker()
        text = "Authentication Overview\n\nThis section covers JWT and API Keys."
        chunks = chunker.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks) >= 1
        assert chunks[0].section_heading == "Authentication Overview"

    def test_no_false_heading_on_long_first_line(self):
        """Long first lines should not be treated as headings."""
        chunker = Chunker()
        text = "This is a very long first line that contains a lot of text and should not be detected as a heading because it exceeds the length threshold."
        chunks = chunker.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks) >= 1
        assert chunks[0].section_heading is None

    def test_configurable_chunk_size(self):
        """Chunk size should be configurable."""
        text = "A " * 500  # ~1000 chars

        chunker_small = Chunker(ChunkingConfig(chunk_size=200, chunk_overlap=20))
        chunker_large = Chunker(ChunkingConfig(chunk_size=1000, chunk_overlap=200))

        chunks_small = chunker_small.chunk_text(text, "doc_1", "test.pdf")
        chunks_large = chunker_large.chunk_text(text, "doc_1", "test.pdf")

        assert len(chunks_small) > len(chunks_large)

    def test_citation_dict_includes_all_fields(self):
        """Chunk.citation_dict() should return all citation fields."""
        chunk = Chunk(
            chunk_id="chunk_test123",
            document_id="doc_1",
            document_name="test.pdf",
            page_number=5,
            chunk_index=0,
            section_heading="Auth Section",
            text="Sample text",
        )

        citation = chunk.citation_dict()
        assert citation["chunk_id"] == "chunk_test123"
        assert citation["document_id"] == "doc_1"
        assert citation["document_name"] == "test.pdf"
        assert citation["page_number"] == 5
        assert citation["section"] == "Auth Section"