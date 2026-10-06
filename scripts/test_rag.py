#!/usr/bin/env python3
"""End-to-end RAG pipeline test.

Usage:
    python3 scripts/test_rag.py "What authentication mechanism does the API use?"
    python3 scripts/test_rag.py "How do API keys work?" --top-k 3
    python3 scripts/test_rag.py "What database does the company use?" --no-llm
"""

import argparse
import sys
from pathlib import Path

from app.config.settings import settings
from app.rag.generation import RAGService
from app.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def test_rag(
    query: str,
    index_dir: Path,
    top_k: int = 5,
    use_llm: bool = True,
) -> None:
    """Test the RAG pipeline end-to-end.

    Args:
        query: User query string.
        index_dir: Directory containing FAISS index.
        top_k: Number of chunks to retrieve.
        use_llm: Whether to generate answer with Groq.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Initialize RAG service
    try:
        rag = RAGService(
            index_dir=str(index_dir),
            top_k=top_k,
            temperature=0.1,
            max_tokens=500,
        )
    except FileNotFoundError:
        print(f"\nIndex not found at {index_dir}.")
        print("Run 'python3 scripts/index.py' first to create the index.")
        sys.exit(1)

    if not use_llm:
        print("\n[Retrieval only mode - skipping LLM generation]")
        # Just test retrieval
        from app.rag.embeddings import SentenceTransformerEmbeddingProvider
        from app.rag.retrieval import FAISSRetriever
        from app.rag.vectorstore import FAISSVectorStore

        embedder = SentenceTransformerEmbeddingProvider()
        vector_store = FAISSVectorStore(dimension=embedder.dimension)
        retriever = FAISSRetriever(vector_store=vector_store, embedder=embedder)
        vector_store.load(str(index_dir))

        chunks = retriever.retrieve(query, k=top_k)
        print(f"\nRetrieved {len(chunks)} chunks:\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"  [{i}] {chunk.document_name} — page {chunk.page_number}")
            print(f"      section: {chunk.section_heading or 'N/A'}")
            print(f"      score: {chunk.score:.4f}")
            print(f"      text: {chunk.text[:200]}...")
            print()
        return

    # Run full RAG pipeline
    result = rag.answer_question(query)

    print(f"\nAnswer:")
    print(f"{result.answer}")

    print(f"\nSources ({len(result.citations)}):")
    for i, citation in enumerate(result.citations, 1):
        section_str = f" ({citation.section})" if citation.section else ""
        print(f"  [{i}] {citation.document_name} — Page {citation.page_number}{section_str}")

    print(f"\nMetadata:")
    print(f"  Retrieved chunks: {result.retrieved_chunks_count}")
    print(f"  Processing time: {result.processing_time_ms}ms")
    if result.retrieval_scores:
        print(f"  Retrieval scores: {[f'{s:.3f}' for s in result.retrieval_scores]}")


def main():
    parser = argparse.ArgumentParser(description="Test RAG pipeline end-to-end")
    parser.add_argument("query", type=str, help="Query to test")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=settings.data_index_dir,
        help="Directory containing FAISS index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.default_top_k,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation, only test retrieval",
    )

    args = parser.parse_args()

    if not (args.index_dir / "index.faiss").exists():
        print(f"Index not found at {args.index_dir}. Run 'python3 scripts/index.py' first.")
        sys.exit(1)

    setup_logging()
    test_rag(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()