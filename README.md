# Enterprise Knowledge Assistant

Multi-Agent RAG System for Enterprise Document Q&A

## Overview

This is a portfolio project demonstrating a multi-agent RAG (Retrieval-Augmented Generation) system built with:
- LangGraph for workflow orchestration
- OpenAI for embeddings and LLM
- FAISS for vector search
- FastAPI for REST API
- Streamlit for UI

## Quick Start

```bash
# Install dependencies
python3 -m pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Create sample PDF for testing
python3 scripts/create_sample_pdf.py

# Run FastAPI server
python3 -m uvicorn app.main:app --reload

# Test health endpoint
curl http://localhost:8000/health
```

## Project Structure

```
enterprise-knowledge-assistant/
├── app/                    # Core application
│   ├── config/            # Settings
│   ├── models/            # Pydantic models
│   ├── rag/               # RAG pipeline components
│   ├── graph/             # LangGraph workflow
│   ├── prompts/           # Agent prompts
│   ├── api/               # FastAPI routes
│   └── utils/             # Utilities
├── streamlit/             # Streamlit UI
├── scripts/               # Utility scripts
├── tests/                 # Tests
├── data/                  # Data directories
└── docs/                  # Documentation
```

## Development

```bash
# Run tests
python3 -m pytest

# Lint
python3 -m ruff check .

# Type check
python3 -m mypy app/
```