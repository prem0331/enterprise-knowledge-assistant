from enum import StrEnum

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    """Classification of user query intent."""
    TECHNICAL = "technical"
    POLICY = "policy"
    TROUBLESHOOTING = "troubleshooting"
    GENERAL = "general"
    UNKNOWN = "unknown"


class QueryRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: str | None = Field(None, description="Optional session identifier")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")


class RouterOutput(BaseModel):
    """Output from the router agent."""
    rewritten_query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    retrieval_required: bool = True
    reasoning: str = ""
