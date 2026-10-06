import uuid

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk with full source metadata for citations."""
    chunk_id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}")
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    section_heading: str | None = None
    text: str

    # For retrieval scoring
    score: float | None = None

    def citation_dict(self) -> dict:
        """Return citation information for this chunk."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section": self.section_heading,
        }

    def __str__(self) -> str:
        return f"[{self.document_name} p.{self.page_number}] {self.text[:100]}..."
