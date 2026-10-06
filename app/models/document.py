import hashlib
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for an uploaded document."""
    document_id: str
    filename: str
    file_hash: str
    page_count: int
    file_size_bytes: int
    upload_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    content_type: str = "application/pdf"


class Document(BaseModel):
    """Represents an uploaded document with its metadata and extracted pages."""
    metadata: DocumentMetadata
    pages: list[str]  # Text content per page

    @classmethod
    def create(cls, filename: str, content: bytes, pages: list[str]) -> "Document":
        """Create a document from raw PDF content and extracted pages."""
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        document_id = f"doc_{file_hash}"
        return cls(
            metadata=DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_hash=file_hash,
                page_count=len(pages),
                file_size_bytes=len(content),
            ),
            pages=pages,
        )
