"""Sentence Transformer embedding provider.

Uses sentence-transformers library for local embedding generation.
No API key required - runs entirely on local hardware.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.rag.embeddings.interface import EmbeddingProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Reduce sentence-transformers logging noise
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using Sentence Transformers.

    Uses all-MiniLM-L6-v2 by default (384 dimensions, fast, good quality).
    Runs entirely locally - no API calls, no API key needed.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """Initialize the embedding provider.

        Args:
            model_name: Sentence Transformer model name.
                       Defaults to settings.embedding_model.
            device: Device to run on ('cpu', 'cuda', 'mps').
                    Defaults to auto-detect.
        """
        self._model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._device = device
        self._dimension: int = 0  # Will be set on first load

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension == 0:
            # Lazy load to get dimension
            self._ensure_model_loaded()
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info(
                "Loading Sentence Transformer model",
                extra={"model": self._model_name, "device": self._device},
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            # Get actual dimension from model
            dim = self._model.get_sentence_embedding_dimension()
            self._dimension = dim if dim is not None else 0
            logger.info(
                "Model loaded",
                extra={"model": self._model_name, "dimension": self._dimension},
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._ensure_model_loaded()
        assert self._model is not None

        logger.debug(
            "Embedding texts",
            extra={"count": len(texts), "model": self._model_name},
        )

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize separately for FAISS
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Args:
            query: Query text to embed.

        Returns:
            1D numpy array of shape (dimension,).
        """
        result = self.embed_texts([query])
        return np.array(result[0], dtype=np.float32)
