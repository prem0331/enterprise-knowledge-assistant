"""Context builder for RAG pipeline.

Formats retrieved chunks into a structured context string for the LLM.
Each chunk includes clear source boundaries and metadata for citations.
"""

from app.models.chunk import Chunk


def build_context(chunks: list[Chunk], max_chars: int = 8000) -> tuple[str, list[dict]]:
    """Build context string from retrieved chunks.

    Args:
        chunks: List of retrieved Chunk objects with metadata and scores.
        max_chars: Maximum total characters for context (truncates if needed).

    Returns:
        Tuple of (context_string, citation_metadata_list)
    """
    if not chunks:
        return "No relevant context found.", []

    context_parts = []
    citations = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        source_header = (
            f"[Source {i}]\n"
            f"Document: {chunk.document_name}\n"
            f"Page: {chunk.page_number}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Section: {chunk.section_heading or 'N/A'}\n"
            f"Content:\n{chunk.text}\n"
        )

        # Check if adding this chunk would exceed max_chars
        if total_chars + len(source_header) > max_chars:
            # Truncate the chunk text to fit
            remaining = max_chars - total_chars - 200  # Reserve space for header
            if remaining > 100:
                truncated = f"{source_header[:remaining]}... [truncated]\n"
                context_parts.append(truncated)
                citations.append({
                    "source_number": i,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section_heading,
                    "truncated": True,
                })
            break

        context_parts.append(source_header)
        citations.append({
            "source_number": i,
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "section": chunk.section_heading,
        })
        total_chars += len(source_header)

    context = "\n---\n".join(context_parts)
    return context, citations


def format_chunk_for_citation(chunk: Chunk, source_number: int) -> dict:
    """Create a citation dict from a chunk.

    Args:
        chunk: Chunk object with metadata.
        source_number: 1-based index in retrieval results.

    Returns:
        Citation dictionary.
    """
    return {
        "source_number": source_number,
        "document_name": chunk.document_name,
        "page_number": chunk.page_number,
        "chunk_id": chunk.chunk_id,
        "section": chunk.section_heading,
    }
