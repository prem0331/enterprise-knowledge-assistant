#!/usr/bin/env python3
"""Index documents for the RAG system.

Usage:
    python scripts/index.py                    # Index all PDFs in data/uploads
    python scripts/index.py --pdf path.pdf    # Index specific PDF
    python scripts/index.py --clear           # Clear existing index
"""

import argparse
import sys
from pathlib import Path

from app.config.settings import settings
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.ingestion import PDFProcessor, Chunker, ChunkingConfig
from app.rag.retrieval import FAISSRetriever
from app.rag.vectorstore import FAISSVectorStore
from app.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def index_documents(
    pdf_paths: list[Path],
    index_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    clear: bool = False,
) -> dict:
    """Index a list of PDF documents.

    Args:
        pdf_paths: List of PDF file paths to index.
        index_dir: Directory to save FAISS index.
        chunk_size: Chunk size in characters.
        chunk_overlap: Chunk overlap in characters.
        clear: Whether to clear existing index first.

    Returns:
        Dictionary with indexing statistics.
    """
    # Initialize components
    processor = PDFProcessor()
    chunker = Chunker(ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    embedder = SentenceTransformerEmbeddingProvider()
    vector_store = FAISSVectorStore(dimension=embedder.dimension)

    # Load existing index if not clearing
    if not clear and (index_dir / "index.faiss").exists():
        logger.info("Loading existing index", extra={"path": str(index_dir)})
        vector_store.load(str(index_dir))
    elif clear:
        logger.info("Clearing existing index")

    total_chunks = 0
    indexed_count = 0

    for pdf_path in pdf_paths:
        logger.info("Processing document", extra={"path": str(pdf_path)})

        # Extract document
        document = processor.extract_document(pdf_path)

        # Chunk document
        chunks = chunker.chunk_document(document)

        if not chunks:
            logger.warning("No chunks generated", extra={"path": str(pdf_path)})
            continue

        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        logger.info("Generating embeddings", extra={"count": len(texts)})
        embeddings = embedder.embed_and_normalize(texts)

        # Add to vector store
        vector_store.add_documents(chunks, embeddings)

        total_chunks += len(chunks)
        indexed_count += 1

    # Save index
    logger.info("Saving index", extra={"path": str(index_dir)})
    vector_store.save(str(index_dir))

    stats = {
        "indexed_count": indexed_count,
        "total_chunks": total_chunks,
        "dimension": embedder.dimension,
        "model": embedder.model_name,
    }
    logger.info("Indexing complete", extra=stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Specific PDF file to index",
    )
    parser.add_argument(
        "--upload-dir",
        type=Path,
        default=settings.data_uploads_dir,
        help="Directory containing PDFs to index",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.data_index_dir,
        help="Directory to save FAISS index",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Chunk overlap in characters",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before indexing",
    )

    args = parser.parse_args()

    # Determine PDF files to process
    if args.pdf:
        pdf_paths = [args.pdf]
    else:
        pdf_paths = list(args.upload_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDF files found", extra={"dir": str(args.upload_dir)})
            sys.exit(0)

    # Run indexing
    try:
        stats = index_documents(
            pdf_paths=pdf_paths,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            clear=args.clear,
        )
        print(f"\nIndexing complete!")
        print(f"  Documents indexed: {stats['indexed_count']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Embedding model: {stats['model']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Index saved to: {args.index_dir}")
    except Exception as e:
        logger.error("Indexing failed", extra={"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    main()