"""LangGraph workflow compilation for the RAG pipeline.

Defines the graph structure with nodes and conditional edges.
"""


from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    context_node,
    decide_node,
    generate_node,
    ground_node,
    increment_retry_node,
    retrieve_node,
    router_node,
)
from app.graph.state import GraphState
from app.utils.logging import get_logger

logger = get_logger(__name__)


def create_workflow() -> CompiledStateGraph[GraphState, Any, GraphState, GraphState]:
    """Create and compile the LangGraph workflow.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    # Create graph with state schema
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("context", context_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("ground", ground_node)
    workflow.add_node("increment_retry", increment_retry_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    # Router -> Retrieve (conditional on retrieval_required)
    workflow.add_conditional_edges(
        "router",
        lambda state: "retrieve" if state.retrieval_required else "generate",
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # Retrieve -> Context
    workflow.add_edge("retrieve", "context")

    # Context -> Generate
    workflow.add_edge("context", "generate")

    # Generate -> Ground
    workflow.add_edge("generate", "ground")

    # Ground -> Decide (conditional)
    workflow.add_conditional_edges(
        "ground",
        decide_node,
        {
            "retrieve": "retrieve",
            "generate": "increment_retry",
            END: END,
        },
    )

    # Increment retry -> Generate
    workflow.add_edge("increment_retry", "generate")

    # Compile
    app: CompiledStateGraph[GraphState, Any, GraphState, GraphState] = workflow.compile()

    logger.info("LangGraph workflow compiled successfully")
    return app


def run_rag_workflow(
    query: str,
    session_id: str | None = None,
    top_k: int = 5,
) -> GraphState:
    """Run the RAG workflow for a single query.

    Args:
        query: User question.
        session_id: Optional session identifier.
        top_k: Number of chunks to retrieve.

    Returns:
        Final GraphState with all results.
    """
    app = create_workflow()

    initial_state = GraphState(
        original_query=query,
        session_id=session_id,
        top_k=top_k,
    )

    result_dict = app.invoke(initial_state)
    # Convert dict back to GraphState
    return GraphState(**result_dict)


if __name__ == "__main__":
    # Quick test
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else (
        "What authentication mechanism does the API use?"
    )
    result = run_rag_workflow(query)
    print(f"Answer: {result.answer}")
    print(
        f"Grounding passed: {result.grounding_result.passed if result.grounding_result else 'N/A'}"
    )
    print(f"Citations: {len(result.citations)}")
    print(f"Retries: {result.retry_count}")
