"""FAISS vector store implementation.

Uses IndexFlatIP (inner product) with L2-normalized embeddings
for cosine similarity search.

Persists:
- FAISS index file (.faiss)
- Metadata JSON file (chunk_id -> Chunk metadata mapping)
"""

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.models.chunk import Chunk
from app.rag.vectorstore.interface import VectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store with metadata persistence.

    Uses IndexFlatIP for exact inner product search.
    Embeddings must be L2-normalized before adding/searching
    for cosine similarity semantics.

    Stores metadata in a separate JSON file for efficient lookup.
    """

    def __init__(self, dimension: int):
        """Initialize the FAISS vector store.

        Args:
            dimension: Embedding dimension.
        """
        self.dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: dict[str, Chunk] = {}  # chunk_id -> Chunk

    @property
    def index(self) -> faiss.IndexFlatIP | None:
        """Get the FAISS index."""
        return self._index

    def _ensure_index(self) -> None:
        """Create index if not exists."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dimension)

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

        Raises:
            ValueError: If embeddings shape doesn't match chunks or dimension.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Number of chunks ({len(chunks)}) doesn't match "
                f"embeddings shape ({embeddings.shape[0]})"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) doesn't match "
                f"store dimension ({self.dimension})"
            )

        self._ensure_index()
        assert self._index is not None

        # Add to FAISS index
        self._index.add(embeddings)

        # Store metadata
        for chunk in chunks:
            self._metadata[chunk.chunk_id] = chunk

        logger.info(
            "Added documents to FAISS",
            extra={
                "count": len(chunks),
                "total_vectors": self._index.ntotal,
            },
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[Chunk]:
        """Search for similar chunks using inner product (cosine similarity).

        Args:
            query_embedding: 1D numpy array of shape (dimension,).
                            Must be L2-normalized.
            k: Number of results to return.

        Returns:
            List of Chunk objects with score attribute set to similarity score.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be 1D array")
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.shape[0]}) "
                f"doesn't match store dimension ({self.dimension})"
            )

        # Reshape to (1, dimension) for FAISS
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Search
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        # Map results to chunks
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots

            # Get chunk_id from index position
            chunk_id = self._get_chunk_id_by_index(idx)
            if chunk_id and chunk_id in self._metadata:
                chunk = self._metadata[chunk_id].model_copy()
                chunk.score = float(score)
                results.append(chunk)

        logger.debug(
            "FAISS search complete",
            extra={"k": k, "results": len(results)},
        )
        return results

    def _get_chunk_id_by_index(self, idx: int) -> str | None:
        """Get chunk_id by FAISS index position.

        Since we store chunks in order, we can maintain a list.
        For now, we'll use the metadata keys which should be in insertion order.
        """
        chunk_ids = list(self._metadata.keys())
        if 0 <= idx < len(chunk_ids):
            return chunk_ids[idx]
        return None

    def save(self, path: str) -> None:
        """Persist the index and metadata to disk atomically.

        Args:
            path: Directory path to save to.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Write to temporary files first, then atomic rename
        index_tmp = save_path / "index.faiss.tmp"
        meta_tmp = save_path / "metadata.json.tmp"
        index_final = save_path / "index.faiss"
        meta_final = save_path / "metadata.json"

        try:
            # Save FAISS index
            if self._index is not None:
                faiss.write_index(self._index, str(index_tmp))

            # Save metadata
            metadata_dict = {
                chunk_id: chunk.model_dump()
                for chunk_id, chunk in self._metadata.items()
            }
            with open(meta_tmp, "w") as f:
                json.dump(metadata_dict, f)

            # Atomic rename
            os.replace(index_tmp, index_final)
            os.replace(meta_tmp, meta_final)

            logger.info(
                "FAISS index saved",
                extra={
                    "path": str(save_path),
                    "vectors": self._index.ntotal if self._index else 0,
                    "chunks": len(self._metadata),
                },
            )
        except Exception:
            # Cleanup on failure
            for tmp in [index_tmp, meta_tmp]:
                if tmp.exists():
                    tmp.unlink()
            raise

    def load(self, path: str) -> None:
        """Load the index and metadata from disk.

        Args:
            path: Directory path to load from.
        """
        load_path = Path(path)
        index_file = load_path / "index.faiss"
        meta_file = load_path / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index files not found in {path}")

        # Load FAISS index
        loaded_index = faiss.read_index(str(index_file))

        # Verify dimension matches
        if loaded_index.d != self.dimension:
            raise ValueError(
                f"Index dimension ({loaded_index.d}) doesn't match "
                f"configured dimension ({self.dimension})"
            )

        # Cast to IndexFlatIP since we only use that type
        self._index = loaded_index  # type: ignore[assignment]

        # Load metadata
        with open(meta_file) as f:
            metadata_dict = json.load(f)

        self._metadata = {
            chunk_id: Chunk(**chunk_data)
            for chunk_id, chunk_data in metadata_dict.items()
        }

        logger.info(
            "FAISS index loaded",
            extra={
                "path": str(load_path),
                "vectors": self._index.ntotal if self._index else 0,
                "chunks": len(self._metadata),
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "dimension": self.dimension,
            "total_vectors": self._index.ntotal if self._index else 0,
            "total_chunks": len(self._metadata),
            "index_type": "IndexFlatIP",
        }

    def clear(self) -> None:
        """Clear all data from the store."""
        self._index = None
        self._metadata = {}
        logger.info("FAISS vector store cleared")
