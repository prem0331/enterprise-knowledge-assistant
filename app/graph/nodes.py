"""LangGraph nodes for the RAG workflow.

Each node is a function that takes the current state and returns
an updated state dictionary.
"""

from langgraph.graph import END

from app.config.settings import settings
from app.graph.state import GraphState
from app.models.query import QueryIntent
from app.models.response import Citation, GroundingResult
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.generation.context_builder import build_context
from app.rag.llm import GroqLLMProvider
from app.rag.retrieval import FAISSRetriever
from app.rag.vectorstore import FAISSVectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Shared components (initialized once)
_embedder: SentenceTransformerEmbeddingProvider | None = None
_vector_store: FAISSVectorStore | None = None
_retriever: FAISSRetriever | None = None
_llm: GroqLLMProvider | None = None


def _get_components() -> tuple:
    """Lazy-initialize shared components."""
    global _embedder, _vector_store, _retriever, _llm

    if _embedder is None:
        _embedder = SentenceTransformerEmbeddingProvider()
        _vector_store = FAISSVectorStore(dimension=_embedder.dimension)
        _retriever = FAISSRetriever(vector_store=_vector_store, embedder=_embedder)
        _llm = GroqLLMProvider()

    return _embedder, _vector_store, _retriever, _llm


def _load_index(index_dir: str | None = None) -> None:
    """Ensure FAISS index is loaded."""
    _, vector_store, _, _ = _get_components()
    if vector_store.index is None or vector_store.index.ntotal == 0:
        idx_dir = index_dir or str(settings.data_index_dir)
        vector_store.load(idx_dir)
        logger.info("FAISS index loaded in workflow", extra={"path": idx_dir})


# ===== Node: Router =====
def router_node(state: GraphState) -> dict:
    """Route the query: understand intent, rewrite if needed."""
    logger.info("Router node: processing query", extra={"query": state.original_query[:100]})
    state.update_timestamp()

    # For now, simple routing - in future can use LLM for query rewriting
    # This is a placeholder that can be enhanced with actual LLM call
    query = state.original_query.strip()

    # Simple intent classification based on keywords
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["auth", "login", "token", "jwt", "oauth", "api key"]):
        intent = QueryIntent.TECHNICAL
    elif any(kw in query_lower for kw in ["policy", "compliance", "rule", "regulation"]):
        intent = QueryIntent.POLICY
    elif any(kw in query_lower for kw in ["error", "fail", "issue", "bug", "troubleshoot", "fix"]):
        intent = QueryIntent.TROUBLESHOOTING
    else:
        intent = QueryIntent.GENERAL

    # Determine if retrieval is needed
    retrieval_required = not any(
        kw in query_lower for kw in ["hello", "hi", "thanks", "thank you", "bye"]
    )

    # Query rewriting (simple version - can be enhanced with LLM)
    rewritten_query = query

    logger.info(
        "Router node complete",
        extra={
            "intent": intent.value,
            "retrieval_required": retrieval_required,
            "rewritten_query": rewritten_query[:100],
        },
    )

    routing_reasoning = (
        f"Classified as {intent.value}, retrieval_required={retrieval_required}"
    )
    return {
        "rewritten_query": rewritten_query,
        "intent": intent,
        "retrieval_required": retrieval_required,
        "routing_reasoning": routing_reasoning,
    }


# ===== Node: Retrieve =====
def retrieve_node(state: GraphState) -> dict:
    """Retrieve relevant chunks from FAISS."""
    rewritten = state.rewritten_query
    query_text = rewritten[:100] if rewritten else "none"
    logger.info("Retrieve node: searching", extra={"query": query_text})

    if not state.retrieval_required:
        logger.info("Retrieval not required, skipping")
        return {
            "retrieved_chunks": [],
            "retrieval_query": state.rewritten_query,
            "retrieval_scores": [],
        }

    _load_index(state.session_id)  # session_id not used for index path currently

    _, _, retriever, _ = _get_components()

    query_for_retrieve = state.rewritten_query or state.original_query
    chunks = retriever.retrieve(query_for_retrieve, k=state.top_k)

    scores = [c.score for c in chunks if c.score is not None]

    logger.info(
        "Retrieve node complete",
        extra={"chunks_found": len(chunks), "scores": [f"{s:.3f}" for s in scores]},
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_query": state.rewritten_query,
        "retrieval_scores": scores,
    }


# ===== Node: Context =====
def context_node(state: GraphState) -> dict:
    """Filter chunks and build context package."""
    chunk_count = len(state.retrieved_chunks)
    logger.info("Context node: building context", extra={"chunk_count": chunk_count})

    if not state.retrieved_chunks:
        logger.warning("No chunks to build context from")
        return {
            "filtered_chunks": [],
            "context_package": "No relevant information found in the knowledge base.",
            "missing_info": "No chunks retrieved",
            "needs_reretrieval": False,
        }

    # For now, use all retrieved chunks (can add filtering/reranking later)
    filtered = state.retrieved_chunks

    # Build context
    context, citation_metadata = build_context(filtered)

    # Check if we might need reretrieval (e.g., low scores)
    needs_reretrieval = False
    if state.retrieval_scores and max(state.retrieval_scores) < 0.3:
        needs_reretrieval = True

    logger.info(
        "Context node complete",
        extra={
            "filtered_chunks": len(filtered),
            "context_chars": len(context),
            "needs_reretrieval": needs_reretrieval,
        },
    )

    return {
        "filtered_chunks": filtered,
        "context_package": context,
        "missing_info": None,
        "needs_reretrieval": needs_reretrieval,
    }


# ===== Node: Generate =====
def generate_node(state: GraphState) -> dict:
    """Generate answer using LLM with context."""
    context_pkg = state.context_package
    ctx_chars = len(context_pkg) if context_pkg else 0
    logger.info("Generate node: calling LLM", extra={"context_chars": ctx_chars})

    if not state.context_package or not state.filtered_chunks:
        no_info_msg = (
            "I don't have enough information in the knowledge base "
            "to answer this question."
        )
        return {
            "answer": no_info_msg,
            "citations": [],
            "generation_reasoning": "No context available",
        }

    _, _, _, llm = _get_components()

    # Build citation metadata for context builder format
    citation_metadata = []
    for i, chunk in enumerate(state.filtered_chunks, 1):
        citation_metadata.append({
            "source_number": i,
            "document_name": chunk.document_name,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "section": chunk.section_heading,
        })

    system_prompt = (
        "You are an enterprise knowledge assistant. Answer the user's question "
        "using ONLY the provided context. If the context does not contain enough "
        "information to answer the question, explicitly state that the available "
        "documents do not provide enough information. Do not invent information. "
        "Cite sources using [Source X] format where X corresponds to the source "
        "number in the context. Keep your answer concise and useful."
    )

    user_prompt = (
        f"Question: {state.original_query}\n\n"
        f"Context:\n{state.context_package}\n\n"
        "Answer:"
    )

    try:
        answer = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=500,
        )
    except Exception as e:
        logger.error("LLM generation failed", extra={"error": str(e)})
        answer = f"Error generating answer: {str(e)}"

    # Build citation objects
    citations: list[Citation] = []
    for meta in citation_metadata:
        doc_name = str(meta["document_name"])
        page_num = int(str(meta["page_number"]))
        chunk_id = str(meta["chunk_id"])
        section = str(meta["section"]) if meta["section"] else None
        citations.append(Citation(
            document_name=doc_name,
            page_number=page_num,
            chunk_id=chunk_id,
            section=section,
            text_snippet="",
        ))

    ans_chars = len(answer)
    cit_count = len(citations)
    logger.info("Generate node complete", extra={"answer_chars": ans_chars, "citations": cit_count})

    return {
        "answer": answer,
        "citations": citations,
        "generation_reasoning": "Generated with Groq LLM using retrieved context",
    }


# ===== Node: Ground =====
def ground_node(state: GraphState) -> dict:
    """Validate that answer is grounded in retrieved context."""
    ans_chars = len(state.answer) if state.answer else 0
    logger.info("Ground node: validating answer", extra={"answer_chars": ans_chars})

    if not state.answer or not state.filtered_chunks:
        logger.warning("No answer or chunks to ground")
        return {
            "grounding_result": GroundingResult(
                passed=False,
                reason="No answer or context to validate",
                unsupported_claims=["No answer generated"],
                verified_citations=[],
            ),
        }

    # Simple grounding check: verify each citation corresponds to actual chunk
    verified_citations = []
    unsupported_claims = []

    chunk_texts = {c.chunk_id: c.text.lower() for c in state.filtered_chunks}

    for citation in state.citations:
        if citation.chunk_id in chunk_texts:
            verified_citations.append(citation.chunk_id)
        else:
            unsupported_claims.append(f"Citation to missing chunk: {citation.chunk_id}")

    passed = len(unsupported_claims) == 0

    if passed:
        reason = "All citations verified"
    else:
        reason = f"Unsupported claims: {unsupported_claims}"

    grounding_result = GroundingResult(
        passed=passed,
        reason=reason,
        unsupported_claims=unsupported_claims,
        verified_citations=verified_citations,
    )

    logger.info(
        "Ground node complete",
        extra={
            "passed": passed,
            "verified": len(verified_citations),
            "unsupported": len(unsupported_claims),
        },
    )

    return {
        "grounding_result": grounding_result,
    }


# ===== Node: Decide =====
def decide_node(state: GraphState) -> str:
    """Decide next step based on grounding result and retry count."""
    logger.info("Decide node: evaluating", extra={"retry_count": state.retry_count})

    # If grounding passed, we're done
    if state.grounding_result and state.grounding_result.passed:
        logger.info("Grounding passed, ending workflow")
        return END

    # If max retries reached, end with best effort
    if state.retry_count >= state.max_retries:
        logger.warning("Max retries reached, ending workflow")
        return END

    # If needs reretrieval, go back to retrieve
    if state.needs_reretrieval:
        logger.info("Reretrieval needed, going to retrieve")
        return "retrieve"

    # Otherwise, retry generation
    logger.info("Retrying generation")
    return "generate"


# ===== Node: Reretry (increment counter) =====
def increment_retry_node(state: GraphState) -> dict:
    """Increment retry counter and update state."""
    new_count = state.retry_count + 1
    log_extra = {"old_count": state.retry_count, "new_count": new_count}
    logger.info("Incrementing retry", extra=log_extra)
    return {
        "retry_count": new_count,
    }
