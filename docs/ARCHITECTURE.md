# Enterprise Knowledge Assistant — Architecture Document

## 1. Repository Assessment

**Current State:**
- Empty repository (no Python files, no dependencies, no config, no git)
- Only `docs/PROJECT_SPEC.md` and a resume PDF exist
- Fresh start - no legacy code to migrate or refactor

---

## 2. Proposed Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  Upload     │  │   Index     │  │    Chat     │                 │
│  │  Documents  │  │  Documents  │  │  Interface  │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
└─────────┼────────────────┼────────────────┼────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FASTAPI                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │
│  │ POST /documents │  │ POST /documents │  │ POST /chat      │    │
│  │ /upload         │  │ /index          │  │                 │    │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘    │
└───────────┼────────────────────┼────────────────────┼──────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH ORCHESTRATION                        │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Router     │───▶│  Retrieval   │───▶│  Context     │          │
│  │   Agent      │    │  Agent       │    │  Agent       │          │
│  └──────────────┘    └──────────────┘    └──────┬───────┘          │
│                                                  │                  │
│                          ┌───────────────────────┘                  │
│                          ▼                                         │
│                   ┌──────────────┐    ┌──────────────┐            │
│                   │   Answer     │───▶│  Grounding   │            │
│                   │  Generation  │    │  Validator   │            │
│                   └──────────────┘    └──────┬───────┘            │
└──────────────────────────────────────────────┼────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  Ingestion  │  │ Embeddings  │  │  FAISS      │                 │
│  │  (PDF→Text) │  │  (OpenAI)   │  │  Vector     │                 │
│  │  Chunking   │  └──────┬──────┘  │  Store      │                 │
│  └─────────────┘         │         └──────┬──────┘                 │
└──────────────────────────┼────────────────┼────────────────────────┘
                           │                │
                           ▼                ▼
                    ┌─────────────┐  ┌─────────────┐
                    │  Metadata   │  │   Index     │
                    │  Store      │  │  Files      │
                    └─────────────┘  └─────────────┘
```

---

## 3. Directory Structure

```
enterprise-knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   └── health.py
│   │   ├── schemas.py
│   │   └── dependencies.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── context_agent.py
│   │   ├── answer_agent.py
│   │   └── grounding_agent.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── workflow.py
│   │   └── nodes.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_processor.py
│   │   │   ├── chunker.py
│   │   │   └── metadata.py
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── interface.py
│   │   │   └── openai_embeddings.py
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── interface.py
│   │   │   └── faiss_retriever.py
│   │   └── vectorstore/
│   │       ├── __init__.py
│   │       ├── interface.py
│   │       └── faiss_store.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── query.py
│   │   └── response.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_service.py
│   │   └── index_service.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── router_prompt.py
│   │   ├── answer_prompt.py
│   │   └── grounding_prompt.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── file_handling.py
├── streamlit/
│   ├── __init__.py
│   ├── app.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── upload.py
│   │   ├── chat.py
│   │   └── citations.py
│   └── utils.py
├── scripts/
│   ├── __init__.py
│   ├── ingest.py
│   ├── index.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_metadata.py
│   │   ├── test_retrieval.py
│   │   └── test_citations.py
│   ├── integration/
│   │   ├── test_router.py
│   │   ├── test_retrieval_workflow.py
│   │   ├── test_grounding.py
│   │   └── test_api.py
│   └── fixtures/
│       ├── sample_docs/
│       └── eval_dataset.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PROGRESS.md
│   └── PROJECT_SPEC.md
├── data/
│   ├── uploads/
│   ├── index/
│   └── metadata/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## 4. LangGraph Design

### 4.1 State Schema (Typed)

```python
# app/graph/state.py
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from app.models.chunk import Chunk
from app.models.query import QueryIntent
from app.models.response import Answer, Citation, GroundingResult

class GraphState(BaseModel):
    # Input
    original_query: str
    session_id: str
    
    # Router Agent Output
    rewritten_query: Optional[str] = None
    intent: Optional[QueryIntent] = None
    retrieval_required: bool = True
    routing_reasoning: Optional[str] = None
    
    # Retrieval Agent Output
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    retrieval_query: Optional[str] = None
    retrieval_scores: List[float] = Field(default_factory=list)
    
    # Context Agent Output
    filtered_chunks: List[Chunk] = Field(default_factory=list)
    context_package: Optional[str] = None
    missing_info: Optional[str] = None
    needs_reretrieval: bool = False
    
    # Answer Agent Output
    answer: Optional[Answer] = None
    generation_reasoning: Optional[str] = None
    
    # Grounding Agent Output
    grounding_result: Optional[GroundingResult] = None
    
    # Control Flow
    retry_count: int = 0
    max_retries: int = 2
    errors: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
```

### 4.2 Nodes (Agents)

| Node | Type | Responsibility | Input | Output |
|------|------|----------------|-------|--------|
| `router` | Agent | Query understanding, intent classification, query rewriting | `original_query` | `rewritten_query`, `intent`, `retrieval_required` |
| `retrieve` | Agent | Semantic search against FAISS, top-k retrieval | `rewritten_query` | `retrieved_chunks`, `retrieval_scores` |
| `context` | Agent | Relevance filtering, context assembly, reretrieval decision | `retrieved_chunks` | `filtered_chunks`, `context_package`, `needs_reretrieval` |
| `generate` | Agent | Answer generation with citations from context | `context_package` | `answer`, `citations` |
| `ground` | Agent | Citation validation, hallucination detection | `answer`, `filtered_chunks` | `grounding_result` |
| `decide` | Conditional | Retry/continue/end based on grounding | `grounding_result`, `retry_count` | Next node |

### 4.3 Edges & Control Flow

```
START → router → [retrieval_required?] → retrieve → context → generate → ground → decide
                                              ↑                    │
                                              │                    ▼
                                              └───── reretrieval ───┘
                                                                      │
                                              ┌───────────────────────┘
                                              ▼
                                    ┌───────────────┐
                                    │   END/ERROR   │
                                    └───────────────┘
```

**Decision Logic in `decide` node:**
- If `grounding_result.passed` → END (success)
- If `grounding_result.passed` is False AND `retry_count < max_retries` → `generate` (regenerate)
- If `grounding_result.passed` is False AND `retry_count >= max_retries` → END (return best effort with warning)
- If `context.needs_reretrieval` AND `retry_count < max_retries` → `retrieve` (reretrieve)

---

## 5. RAG Pipeline Design

### 5.1 Ingestion Flow

```
PDF File
    │
    ▼
┌─────────────────────┐
│  PDF Processor      │  → Extract text per page using pdfplumber/PyMuPDF
│  (pdfplumber)       │     Preserve page numbers, tables, structure
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Document Metadata  │  → document_id, filename, page_count, 
│  Extractor          │     upload_timestamp, file_hash
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Chunker            │  → Semantic chunking (heading-aware)
│  (Recursive)        │     chunk_size=1000, overlap=200
│                     │     Preserve: page_number, section_heading
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Embeddings         │  → text-embedding-3-small (1536 dim)
│  (OpenAI)           │     Batch processing, retry logic
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FAISS Vector Store │  → IndexFlatIP (cosine similarity)
│  + Metadata Store   │     Persist: index.faiss + metadata.json
└─────────────────────┘
```

### 5.2 Chunking Strategy

**RecursiveCharacterTextSplitter with heading awareness:**
- Primary split: Markdown headers (`#`, `##`, `###`)
- Secondary split: Paragraphs (`\n\n`)
- Tertiary split: Sentences
- Chunk size: 1000 tokens (configurable)
- Overlap: 200 tokens (configurable)
- **Metadata preserved per chunk:** `document_id`, `document_name`, `page_number`, `chunk_index`, `section_heading`, `source`

### 5.3 Embeddings Abstraction

```python
# app/rag/embeddings/interface.py
from abc import ABC, abstractmethod
import numpy as np

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray: ...
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray: ...
    
    @property
    @abstractmethod
    def dimension(self) -> int: ...
    
    @property
    @abstractmethod
    def model_name(self) -> str: ...
```

**Implementation:** `OpenAIEmbeddings` using `text-embedding-3-small` (1536 dims, cost-effective)

### 5.4 Vector Store Abstraction

```python
# app/rag/vectorstore/interface.py
from abc import ABC, abstractmethod
from app.models.chunk import Chunk

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, chunks: List[Chunk], embeddings: np.ndarray) -> None: ...
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Chunk]: ...
    
    @abstractmethod
    def save(self, path: str) -> None: ...
    
    @abstractmethod
    def load(self, path: str) -> None: ...
    
    @abstractmethod
    def get_stats(self) -> dict: ...
```

**Implementation:** `FAISSVectorStore` using `IndexFlatIP` with separate JSON metadata file

### 5.5 Retrieval

```python
# app/rag/retrieval/faiss_retriever.py
class FAISSRetriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingProvider):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        query_emb = self.embedder.embed_query(query)
        return self.vector_store.search(query_emb, k=k)
```

### 5.6 Citations Format

```json
{
  "answer": "The API uses JWT authentication...",
  "citations": [
    {
      "document_id": "doc_123",
      "document_name": "api_auth.pdf",
      "page_number": 12,
      "chunk_id": "chunk_456",
      "section": "Authentication",
      "text_snippet": "JWT tokens are used for API authentication..."
    }
  ]
}
```

---

## 6. API Design

### 6.1 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload PDF files |
| POST | `/documents/index` | Trigger indexing of uploaded docs |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{doc_id}` | Remove document from index |
| POST | `/chat` | Ask a question |
| GET | `/chat/history` | Get conversation history |

### 6.2 Request/Response Schemas

```python
# POST /chat
Request:
{
  "query": "What authentication mechanism does our API use?",
  "session_id": "optional-session-id",
  "top_k": 5
}

Response:
{
  "answer": "The API uses JWT authentication...",
  "citations": [
    {
      "document_id": "doc_123",
      "document_name": "api_auth.pdf",
      "page_number": 12,
      "chunk_id": "chunk_456",
      "section": "Authentication",
      "text_snippet": "JWT tokens are used for API authentication..."
    }
  ],
  "retrieved_chunks_count": 5,
  "grounding_passed": true,
  "processing_time_ms": 1250
}

# POST /documents/upload
Request: multipart/form-data with file
Response: { "document_id": "...", "filename": "...", "status": "uploaded" }

# POST /documents/index
Request: { "document_ids": ["doc_123"] }  # or empty for all
Response: { "indexed_count": 1, "total_chunks": 45, "status": "completed" }
```

---

## 7. Streamlit Architecture

### 7.1 Page Structure

```
streamlit/app.py
├── Sidebar
│   ├── Document Upload Section
│   ├── Document List / Index Status
│   └── Settings (top_k, model selection)
├── Main Area
│   ├── Chat History
│   ├── Query Input
│   └── Answer Display
│       ├── Answer Text
│       ├── Citations Panel (expandable)
│       └── Retrieved Context (debug view)
```

### 7.2 Key Components

- **Upload Component**: Drag-drop PDF, show progress, validate file type/size
- **Index Component**: Trigger indexing, show progress, display stats
- **Chat Component**: Streaming response, citation rendering, context viewer
- **Citation Component**: Clickable citations linking to source preview

---

## 8. Implementation Phases

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| **0** | Architecture & Setup | This document, `pyproject.toml`, `.env.example`, `.gitignore`, config |
| **1** | Foundation | Config, logging, models, basic project structure, runnable FastAPI |
| **2** | Document Ingestion | PDF→text→chunks with metadata, unit tests |
| **3** | Embeddings + FAISS | Embedding abstraction, FAISS store, persistence, manual retrieval test |
| **4** | Basic RAG | Query→Retrieve→Generate (single chain, no LangGraph yet) |
| **5** | LangGraph Workflow | State machine with all nodes, conditional edges, observability |
| **6** | Multi-Agent Behavior | Specialized prompts per agent, routing logic, reretrieval |
| **7** | Grounding + Citations | Citation tracking, grounding validator, retry logic |
| **8** | FastAPI | All endpoints, validation, error handling, middleware |
| **9** | Streamlit | Full UI with upload, chat, citations, debug view |
| **10** | Testing + Evaluation | Unit/integration tests, eval dataset, eval script |
| **11** | Production Hardening | Security, performance, error handling, logging |
| **12** | Documentation | README, architecture docs, API docs, limitations |

---

## 9. Important Design Decisions & Tradeoffs

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| **Vector Store** | FAISS (IndexFlatIP) | Simple, fast, no external deps, good for <100k vectors | Not distributed, no horizontal scaling |
| **Embeddings** | OpenAI `text-embedding-3-small` | Good quality/speed/cost, easy API | Requires API key, external dependency |
| **Chunking** | Recursive (heading-aware) | Preserves document structure | More complex than fixed-size |
| **LLM** | OpenAI GPT-4o-mini | Cost-effective, good reasoning | External API dependency |
| **Orchestration** | LangGraph | Explicit state, observability, cycles | Learning curve, more verbose |
| **Citations** | Chunk-level with metadata | Precise, traceable | More storage, complex generation |
| **Grounding** | Separate validator agent | Catches hallucinations explicitly | Extra latency, extra LLM call |
| **Config** | Pydantic Settings + `.env` | Type-safe, validated, 12-factor | Slight boilerplate |

---

## 10. Technical Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| FAISS index corruption | High | Medium | Atomic writes, backup before overwrite, validation on load |
| OpenAI API failures | High | Medium | Retry with exponential backoff, circuit breaker, graceful degradation |
| PDF extraction failures | Medium | High | Try multiple libraries (pdfplumber → PyMuPDF), skip bad pages |
| Hallucination despite grounding | High | Medium | Strict system prompts, citation validation, "I don't know" training |
| Large document handling | Medium | Medium | Streaming upload, chunked processing, progress tracking |
| Token limit exceeded | Medium | Medium | Chunk truncation, context window management, summarize long context |

---

## 11. First Implementation Step (Phase 0 → 1)

**Smallest vertical slice:**
1. Initialize git repo
2. Create `pyproject.toml` with dependencies
3. Create `.env.example` and `.gitignore`
4. Create `app/config/settings.py` with Pydantic Settings
5. Create basic `app/models/` (Document, Chunk, Query, Response)
6. Create `app/utils/logging.py` with structured logging
7. Create minimal FastAPI app with `/health` endpoint
8. Verify: `uvicorn app.main:app --reload` works

This establishes the foundation that all subsequent phases build on.