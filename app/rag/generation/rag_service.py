"""RAG service for end-to-end question answering.

Combines retrieval, context building, and LLM generation.
Simple function chain - no LangGraph yet.
"""

import time
from dataclasses import dataclass

from app.config.settings import settings
from app.models.response import Citation
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.generation.context_builder import build_context
from app.rag.llm import GroqLLMProvider
from app.rag.retrieval import FAISSRetriever
from app.rag.vectorstore import FAISSVectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """Result of the RAG pipeline."""
    answer: str
    citations: list[Citation]
    retrieved_chunks_count: int
    processing_time_ms: int
    retrieval_scores: list[float]


class RAGService:
    """Simple RAG pipeline: Query → Embed → Retrieve → Build Context → Generate Answer."""

    def __init__(
        self,
        index_dir: str | None = None,
        top_k: int | None = None,
        groq_model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ):
        """Initialize the RAG service.

        Args:
            index_dir: Path to FAISS index directory. Defaults to settings.
            top_k: Number of chunks to retrieve. Defaults to settings.
            groq_model: Groq model name. Defaults to settings.
            temperature: LLM temperature.
            max_tokens: Maximum tokens for generation.
        """
        self.index_dir = index_dir or str(settings.data_index_dir)
        self.top_k = top_k or settings.default_top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize components
        self.embedder = SentenceTransformerEmbeddingProvider()
        self.vector_store = FAISSVectorStore(dimension=self.embedder.dimension)
        self.retriever = FAISSRetriever(vector_store=self.vector_store, embedder=self.embedder)
        self.llm = GroqLLMProvider(model=groq_model)

        # Load index on initialization
        self._load_index()

    def _load_index(self) -> None:
        """Load FAISS index from disk."""
        try:
            self.vector_store.load(self.index_dir)
            stats = self.vector_store.get_stats()
            logger.info("RAG service initialized", extra=stats)
        except FileNotFoundError:
            logger.warning(
                "Index not found, run indexing first",
                extra={"path": self.index_dir}
            )
            raise

    def answer_question(self, query: str) -> RAGResult:
        """Run the full RAG pipeline for a query.

        Args:
            query: User question.

        Returns:
            RAGResult with answer, citations, and metadata.
        """
        start_time = time.perf_counter()

        logger.info("Processing query", extra={"query": query[:100]})

        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(query, k=self.top_k)

        if not chunks:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info("No chunks retrieved")
            no_info_msg = (
                "I don't have enough information in the knowledge base "
                "to answer this question."
            )
            return RAGResult(
                answer=no_info_msg,
                citations=[],
                retrieved_chunks_count=0,
                processing_time_ms=elapsed_ms,
                retrieval_scores=[],
            )

        # Step 2: Build context from chunks
        context, citation_metadata = build_context(chunks)

        # Step 3: Generate answer with LLM
        answer = self._generate_answer(query, context)

        # Step 4: Create citation objects
        citations = self._build_citations(citation_metadata, answer)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "RAG pipeline complete",
            extra={
                "query": query[:100],
                "chunks_retrieved": len(chunks),
                "citations": len(citations),
                "processing_time_ms": elapsed_ms,
            },
        )

        return RAGResult(
            answer=answer,
            citations=citations,
            retrieved_chunks_count=len(chunks),
            processing_time_ms=elapsed_ms,
            retrieval_scores=[c.score for c in chunks if c.score is not None],
        )

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq LLM with context.

        Args:
            query: User question.
            context: Formatted context from retrieved chunks.

        Returns:
            Generated answer string.
        """
        system_prompt = (
            "You are an enterprise knowledge assistant. Answer the user's question "
            "using ONLY the provided context. If the context does not contain enough "
            "information to answer the question, explicitly state that the available "
            "documents do not provide enough information. Do not invent information. "
            "Cite sources using [Source X] format where X corresponds to the source "
            "number in the context. Keep your answer concise and useful."
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        return self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _build_citations(
        self,
        citation_metadata: list[dict],
        answer: str,
    ) -> list[Citation]:
        """Build Citation objects from metadata.

        Args:
            citation_metadata: List of citation metadata dicts.
            answer: Generated answer (used for text snippet if needed).

        Returns:
            List of Citation objects.
        """
        citations = []
        for meta in citation_metadata:
            citations.append(Citation(
                document_name=meta["document_name"],
                page_number=meta["page_number"],
                chunk_id=meta["chunk_id"],
                section=meta["section"],
                text_snippet="",  # Could extract relevant snippet from answer
            ))
        return citations


def create_rag_service(**kwargs) -> RAGService:
    """Factory function to create RAGService with default settings."""
    return RAGService(**kwargs)
