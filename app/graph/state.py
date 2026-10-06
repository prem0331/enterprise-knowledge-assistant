"""LangGraph state schema for the RAG workflow.

This defines the complete state that flows through the graph nodes.
Each node reads from and writes to specific fields.
"""

from dataclasses import dataclass, field
from datetime import datetime

from app.models.chunk import Chunk
from app.models.query import QueryIntent
from app.models.response import Citation, GroundingResult


@dataclass
class GraphState:
    """Complete state for the LangGraph RAG workflow.

    Fields are organized by which node produces/consumes them.
    """

    # ===== Input (from user) =====
    original_query: str
    session_id: str | None = None
    top_k: int = 5

    # ===== Router Agent Output =====
    rewritten_query: str | None = None
    intent: QueryIntent | None = None
    retrieval_required: bool = True
    routing_reasoning: str | None = None

    # ===== Retrieval Agent Output =====
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    retrieval_query: str | None = None
    retrieval_scores: list[float] = field(default_factory=list)

    # ===== Context Agent Output =====
    filtered_chunks: list[Chunk] = field(default_factory=list)
    context_package: str | None = None
    missing_info: str | None = None
    needs_reretrieval: bool = False

    # ===== Generation Agent Output =====
    answer: str | None = None
    generation_reasoning: str | None = None
    citations: list[Citation] = field(default_factory=list)

    # ===== Grounding Agent Output =====
    grounding_result: GroundingResult | None = None

    # ===== Control Flow =====
    retry_count: int = 0
    max_retries: int = 2
    errors: list[str] = field(default_factory=list)

    # ===== Metadata =====
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        """Convert state to dictionary for logging/serialization."""
        return {
            "original_query": self.original_query,
            "session_id": self.session_id,
            "top_k": self.top_k,
            "rewritten_query": self.rewritten_query,
            "intent": self.intent.value if self.intent else None,
            "retrieval_required": self.retrieval_required,
            "routing_reasoning": self.routing_reasoning,
            "retrieved_chunks_count": len(self.retrieved_chunks),
            "retrieval_scores": self.retrieval_scores,
            "filtered_chunks_count": len(self.filtered_chunks),
            "needs_reretrieval": self.needs_reretrieval,
            "answer": self.answer,
            "citations_count": len(self.citations),
            "grounding_passed": self.grounding_result.passed if self.grounding_result else None,
            "retry_count": self.retry_count,
            "errors": self.errors,
        }
