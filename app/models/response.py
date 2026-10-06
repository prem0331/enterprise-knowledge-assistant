
from pydantic import BaseModel, Field

from app.models.chunk import Chunk


class Citation(BaseModel):
    """Citation linking answer to source document."""
    document_name: str
    page_number: int
    chunk_id: str
    section: str | None = None
    text_snippet: str = ""

    @classmethod
    def from_chunk(cls, chunk: Chunk, snippet_length: int = 200) -> "Citation":
        return cls(
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            section=chunk.section_heading,
            text_snippet=chunk.text[:snippet_length],
        )


class GroundingResult(BaseModel):
    """Result of grounding validation."""
    passed: bool
    reason: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)
    verified_citations: list[str] = Field(default_factory=list)  # chunk_ids that were verified


class Answer(BaseModel):
    """Generated answer with citations."""
    text: str
    citations: list[Citation] = Field(default_factory=list)
    reasoning: str = ""


class ChatResponse(BaseModel):
    """Complete response from the chat endpoint."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounding_passed: bool
    retrieved_chunks_count: int = 0
    processing_time_ms: int = 0
    session_id: str | None = None
    retry_count: int = 0
