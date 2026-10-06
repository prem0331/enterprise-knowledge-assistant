"""Assistant service - thin wrapper around the LangGraph workflow."""

from pathlib import Path

from app.config.settings import settings
from app.graph.workflow import run_rag_workflow
from app.models.response import ChatResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AssistantService:
    """Service for answering questions using the RAG pipeline."""

    def __init__(
        self,
        index_dir: str | None = None,
        top_k: int = 5,
    ):
        """Initialize the assistant service.

        Args:
            index_dir: Path to FAISS index directory.
            top_k: Number of chunks to retrieve.
        """
        self.index_dir = index_dir or str(settings.data_index_dir)
        self.top_k = top_k

    def answer_question(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int | None = None,
    ) -> ChatResponse:
        """Answer a question using the RAG pipeline.

        Args:
            query: User question.
            session_id: Optional session identifier.
            top_k: Number of chunks to retrieve.

        Returns:
            ChatResponse with answer, citations, and metadata.
        """
        effective_top_k = top_k or self.top_k

        logger.info("Assistant service processing query", extra={
            "query": query[:100],
            "session_id": session_id,
            "top_k": effective_top_k,
        })

        try:
            result = run_rag_workflow(
                query=query,
                session_id=session_id,
                top_k=effective_top_k,
            )

            grounding = result.grounding_result.passed if result.grounding_result else False
            answer_text = result.answer or "No answer generated"
            return ChatResponse(
                answer=answer_text,
                citations=result.citations,
                grounding_passed=grounding,
                retrieved_chunks_count=len(result.retrieved_chunks),
                processing_time_ms=0,
                session_id=session_id,
                retry_count=result.retry_count,
            )
        except Exception as e:
            logger.error("Assistant service error", extra={"error": str(e)})
            raise

    def is_index_ready(self) -> bool:
        """Check if the FAISS index exists and is loadable."""
        index_path = Path(self.index_dir) / "index.faiss"
        meta_path = Path(self.index_dir) / "metadata.json"
        return index_path.exists() and meta_path.exists()


def get_assistant_service() -> AssistantService:
    """Factory function for dependency injection."""
    return AssistantService()
