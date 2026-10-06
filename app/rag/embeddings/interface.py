"""Embedding provider abstraction.

This interface allows swapping embedding implementations without
changing the rest of the application.
"""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (len(texts), dimension).
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query text to embed.

        Returns:
            1D numpy array of shape (dimension,).
        """
        ...

    def embed_and_normalize(self, texts: list[str]) -> np.ndarray:
        """Embed texts and L2-normalize for cosine similarity with IndexFlatIP.

        Args:
            texts: List of text strings to embed.

        Returns:
            Normalized 2D numpy array of shape (len(texts), dimension).
        """
        embeddings = self.embed_texts(texts)
        return self._normalize(embeddings)

    def embed_query_and_normalize(self, query: str) -> np.ndarray:
        """Embed query and L2-normalize.

        Args:
            query: Query text to embed.

        Returns:
            Normalized 1D numpy array of shape (dimension,).
        """
        embedding = self.embed_query(query)
        return self._normalize(embedding.reshape(1, -1)).flatten()

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings for cosine similarity via inner product.

        Args:
            embeddings: 2D numpy array of shape (n, dimension).

        Returns:
            L2-normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
