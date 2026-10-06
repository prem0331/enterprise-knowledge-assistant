from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, health
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    setup_logging()
    logger.info("Application starting up", extra={"version": "0.1.0"})
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="Multi-Agent RAG System for Enterprise Document Q&A",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "name": "Enterprise Knowledge Assistant",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }
