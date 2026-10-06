"""Vector store abstraction.

This interface allows swapping vector store implementations without
changing the rest of the application.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.models.chunk import Chunk


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_documents(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        """Add document chunks with their embeddings.

        Args:
            chunks: List of Chunk objects with metadata.
            embeddings: 2D numpy array of shape (len(chunks), dimension).
                       Must be L2-normalized for IndexFlatIP cosine similarity.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[Chunk]:
        """Search for similar chunks.

        Args:
            query_embedding: 1D numpy array of shape (dimension,).
                            Must be L2-normalized.
            k: Number of results to return.

        Returns:
            List of Chunk objects with score attribute set.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index and metadata to disk.

        Args:
            path: Directory path to save to.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index and metadata from disk.

        Args:
            path: Directory path to load from.
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        ...
