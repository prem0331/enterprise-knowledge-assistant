"""Chat routes for the RAG pipeline."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.models.response import ChatResponse
from app.services.assistant_service import AssistantService, get_assistant_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    top_k: int | None = None


_ASSISTANT_DEP = Depends(get_assistant_service)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    assistant: AssistantService = _ASSISTANT_DEP,
):
    """Process a chat query through the RAG pipeline."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Check if index is ready
    if not assistant.is_index_ready():
        raise HTTPException(
            status_code=404,
            detail="No index found. Please upload and index documents first.",
        )

    try:
        response = assistant.answer_question(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}") from e


@router.get("/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    # For POC, history is stored in Streamlit session state
    # This endpoint is a placeholder
    return {"session_id": session_id, "messages": []}
