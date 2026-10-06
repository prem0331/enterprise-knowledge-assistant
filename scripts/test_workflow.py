#!/usr/bin/env python3
"""Test the LangGraph workflow end-to-end.

Usage:
    python3 scripts/test_workflow.py "What authentication mechanism does the API use?"
    python3 scripts/test_workflow.py "How do API keys work?" --top-k 3
"""

import argparse
import sys
from pathlib import Path

from app.config.settings import settings
from app.graph.workflow import run_rag_workflow
from app.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def test_workflow(
    query: str,
    index_dir: Path,
    top_k: int = 5,
) -> None:
    """Test the LangGraph workflow end-to-end.

    Args:
        query: User query string.
        index_dir: Directory containing FAISS index.
        top_k: Number of chunks to retrieve.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # The workflow loads index internally, just need to ensure it exists
    if not (index_dir / "index.faiss").exists():
        print(f"\nIndex not found at {index_dir}.")
        print("Run 'python3 scripts/index.py' first to create the index.")
        sys.exit(1)

    # Run workflow
    result = run_rag_workflow(
        query=query,
        top_k=top_k,
    )

    print(f"\nAnswer:")
    print(f"{result.answer}")

    print(f"\nSources ({len(result.citations)}):")
    for i, citation in enumerate(result.citations, 1):
        section_str = f" ({citation.section})" if citation.section else ""
        print(f"  [{i}] {citation.document_name} — Page {citation.page_number}{section_str}")

    print(f"\nGrounding: {'PASSED' if result.grounding_result and result.grounding_result.passed else 'FAILED'}")
    if result.grounding_result and not result.grounding_result.passed:
        print(f"  Reason: {result.grounding_result.reason}")
        print(f"  Unsupported: {result.grounding_result.unsupported_claims}")

    print(f"\nMetadata:")
    print(f"  Retrieved chunks: {len(result.retrieved_chunks)}")
    print(f"  Retries: {result.retry_count}")
    print(f"  Intent: {result.intent.value if result.intent else 'N/A'}")
    if result.retrieval_scores:
        print(f"  Retrieval scores: {[f'{s:.3f}' for s in result.retrieval_scores]}")


def main():
    parser = argparse.ArgumentParser(description="Test LangGraph RAG workflow")
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

    args = parser.parse_args()

    if not (args.index_dir / "index.faiss").exists():
        print(f"Index not found at {args.index_dir}. Run 'python3 scripts/index.py' first.")
        sys.exit(1)

    setup_logging()
    test_workflow(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()