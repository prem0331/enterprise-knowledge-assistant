"""FAISS-based retrieval.

Combines embedding provider and vector store for semantic search.
"""

import numpy as np

from app.models.chunk import Chunk
from app.rag.embeddings.interface import EmbeddingProvider
from app.rag.vectorstore.interface import VectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FAISSRetriever:
    """Semantic retriever using FAISS and embeddings."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
    ):
        """Initialize the retriever.

        Args:
            vector_store: Vector store implementation.
            embedder: Embedding provider implementation.
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query string.
            k: Number of chunks to retrieve.

        Returns:
            List of Chunk objects with score and metadata.
        """
        logger.debug("Retrieving chunks", extra={"query": query[:100], "k": k})

        # Embed and normalize query
        query_embedding = self.embedder.embed_query_and_normalize(query)

        # Search
        chunks = self.vector_store.search(query_embedding, k=k)

        logger.info(
            "Retrieval complete",
            extra={"query": query[:100], "k": k, "results": len(chunks)},
        )
        return chunks

    def retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[Chunk]:
        """Retrieve using a pre-computed query embedding.

        Args:
            query_embedding: Pre-computed 1D normalized embedding.
            k: Number of chunks to retrieve.

        Returns:
            List of Chunk objects with score and metadata.
        """
        chunks = self.vector_store.search(query_embedding, k=k)
        return chunks
