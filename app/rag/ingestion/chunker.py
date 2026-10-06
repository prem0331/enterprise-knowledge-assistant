"""Text chunking with configurable recursive splitting.

Uses langchain-text-splitters for reliable, well-tested chunking.
Preserves page-level metadata and attempts to detect section headings.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.chunk import Chunk
from app.models.document import Document
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None

    def __post_init__(self):
        if self.separators is None:
            # Default: paragraphs, then sentences, then characters
            self.separators = ["\n\n", "\n", ". ", " ", ""]


class Chunker:
    """Splits document pages into overlapping chunks with metadata.

    Uses RecursiveCharacterTextSplitter which tries separators in order:
    1. Paragraph breaks (\n\n)
    2. Line breaks (\n)
    3. Sentence endings (. )
    4. Spaces
    5. Characters (fallback)

    This preserves document structure better than fixed-size splitting.
    """

    def __init__(self, config: ChunkingConfig | None = None):
        """Initialize chunker with configuration.

        Args:
            config: ChunkingConfig with chunk_size, chunk_overlap, separators.
        """
        self.config = config or ChunkingConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks with full metadata.

        Args:
            document: Document with pages extracted.

        Returns:
            List of Chunk objects with metadata preserved.
        """
        all_chunks = []
        chunk_index = 0

        for page_num, page_text in enumerate(document.pages, start=1):
            if not page_text.strip():
                logger.debug(
                    "Skipping empty page",
                    extra={"document_id": document.metadata.document_id, "page": page_num},
                )
                continue

            # Split page text into chunks
            page_chunks = self._splitter.split_text(page_text)

            for chunk_text in page_chunks:
                # Try to extract section heading from chunk text
                section = self._extract_section_heading(chunk_text)

                chunk = Chunk(
                    document_id=document.metadata.document_id,
                    document_name=document.metadata.filename,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    section_heading=section,
                    text=chunk_text,
                )
                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "Document chunked",
            extra={
                "document_id": document.metadata.document_id,
                "total_chunks": len(all_chunks),
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            },
        )
        return all_chunks

    def chunk_text(self, text: str, document_id: str, document_name: str,
                   page_number: int = 1, start_chunk_index: int = 0) -> list[Chunk]:
        """Chunk arbitrary text with metadata.

        Useful for testing or non-PDF sources.

        Args:
            text: Text to chunk.
            document_id: Source document ID.
            document_name: Source document name.
            page_number: Page number for citation.
            start_chunk_index: Starting chunk index.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        chunk_texts = self._splitter.split_text(text)
        chunks = []

        for i, chunk_text in enumerate(chunk_texts):
            section = self._extract_section_heading(chunk_text)
            chunks.append(Chunk(
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                chunk_index=start_chunk_index + i,
                section_heading=section,
                text=chunk_text,
            ))

        return chunks

    def _extract_section_heading(self, text: str) -> str | None:
        """Attempt to extract a section heading from chunk text.

        Looks for patterns like "1.2 Section Title" or "## Section Title"
        at the beginning of the chunk. Returns None if not found.

        This is a best-effort heuristic - not guaranteed to find all headings.
        """
        if not text:
            return None

        lines = text.strip().split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()

        # Pattern: "1.2 Section Name" or "1. Section Name" or "## Section Name"
        import re
        heading_patterns = [
            r"^\d+\.\d+\s+(.+)$",      # "1.2 Heading"
            r"^\d+\.\s+(.+)$",          # "1. Heading"
            r"^#{1,3}\s+(.+)$",         # "## Heading" or "### Heading"
        ]

        for pattern in heading_patterns:
            match = re.match(pattern, first_line)
            if match:
                heading = match.group(1).strip()
                if len(heading) < 100:  # Reasonable heading length
                    return heading

        # Check if first line looks like a short heading (under 80 chars, no period at end)
        if len(first_line) < 80 and not first_line.endswith(".") and len(lines) > 1:
            # Heuristic: short first line followed by content
            return first_line

        return None
