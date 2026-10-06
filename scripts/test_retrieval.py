#!/usr/bin/env python3
"""Test retrieval and generation with the RAG system.

Usage:
    python scripts/test_retrieval.py "What authentication mechanism does the API use?"
    python scripts/test_retrieval.py "How do API keys work?" --top-k 3
    python scripts/test_retrieval.py "What are the rate limits?" --no-llm
"""

import argparse
import sys
from pathlib import Path

from app.config.settings import settings
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.llm import GroqLLMProvider
from app.rag.retrieval import FAISSRetriever
from app.rag.vectorstore import FAISSVectorStore
from app.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def test_retrieval(
    query: str,
    index_dir: Path,
    top_k: int = 5,
    use_llm: bool = True,
) -> None:
    """Test retrieval and optionally generate an answer.

    Args:
        query: User query string.
        index_dir: Directory containing FAISS index.
        top_k: Number of chunks to retrieve.
        use_llm: Whether to generate an answer using Groq.
    """
    # Initialize components
    embedder = SentenceTransformerEmbeddingProvider()
    vector_store = FAISSVectorStore(dimension=embedder.dimension)
    retriever = FAISSRetriever(vector_store=vector_store, embedder=embedder)

    # Load index
    logger.info("Loading index", extra={"path": str(index_dir)})
    vector_store.load(str(index_dir))

    stats = vector_store.get_stats()
    print(f"\nIndex stats: {stats}")

    # Retrieve
    print(f"\nQuery: {query}")
    print("-" * 60)

    chunks = retriever.retrieve(query, k=top_k)

    if not chunks:
        print("No relevant chunks found.")
        return

    print(f"\nRetrieved {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk.document_name} — page {chunk.page_number}")
        print(f"      chunk_id: {chunk.chunk_id}")
        print(f"      section: {chunk.section_heading or 'N/A'}")
        print(f"      score: {chunk.score:.4f}")
        print(f"      text: {chunk.text[:200]}...")
        print()

    # Generate answer with Groq if requested
    if use_llm:
        if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
            print("\nSkipping LLM generation: GROQ_API_KEY not configured")
            return

        print("=" * 60)
        print("Generating answer with Groq...")
        print("=" * 60)

        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.document_name}, page {chunk.page_number}]\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are an enterprise knowledge assistant. Answer the user's question "
            "using ONLY the provided context. If the context doesn't contain the answer, "
            "say so. Cite sources using [Source X] format. Do not invent information."
        )

        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        llm = GroqLLMProvider()
        try:
            answer = llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=500,
            )
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            logger.error("LLM generation failed", extra={"error": str(e)})
            print(f"\nLLM generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test RAG retrieval and generation")
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
        print(f"Index not found at {args.index_dir}. Run 'python scripts/index.py' first.")
        sys.exit(1)

    setup_logging()
    test_retrieval(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        use_llm=not args.no_llm,
    )


if __name__ == "__main__":
    main()