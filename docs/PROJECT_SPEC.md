# Enterprise Knowledge Assistant — Multi-Agent RAG System

You are acting as a **Senior AI Engineer and Software Architect** helping me build a production-oriented portfolio project called:

**Enterprise Knowledge Assistant — Multi-Agent RAG System**

I want to build this project incrementally from the existing repository, not as a toy demo and not by generating the entire project blindly in one step.

## Project Goal

Build an enterprise knowledge assistant that allows a user to ask questions about internal enterprise documents such as:

- PDFs
- Technical documentation
- API documentation
- Architecture documents
- Engineering guides
- Policies
- Troubleshooting documents

The system should retrieve relevant information from the uploaded knowledge base and generate answers that are:

- grounded in retrieved documents
- context-aware
- citation-supported
- resistant to hallucination
- traceable to source documents

The core architecture should use:

- Python
- LangGraph
- OpenAI API
- RAG
- FAISS
- semantic search
- document chunking
- FastAPI
- Streamlit

I want the architecture to genuinely justify the term **multi-agent**, rather than simply having multiple functions called agents.

---

# IMPORTANT WORKING RULES

## 1. First inspect the repository

Before writing code:

- inspect the complete repository structure
- inspect existing Python files
- inspect requirements/dependencies
- inspect environment/configuration files
- inspect existing FastAPI code if present
- inspect any existing RAG, LangChain, LangGraph, or OpenAI code
- inspect README/documentation
- identify what can be reused
- identify what needs to be refactored

Do NOT overwrite existing functionality blindly.

At the beginning, give me:

1. Current repository structure
2. What already exists
3. What is missing
4. Proposed architecture
5. Proposed implementation phases
6. Important architectural decisions
7. Risks/tradeoffs

Then wait for my approval before making major architectural changes.

---

# TARGET ARCHITECTURE

Design the system around a LangGraph orchestration layer.

A high-level flow should look similar to:

User Query
   ↓
Query Understanding / Router Agent
   ↓
┌───────────────────────────────┐
│       Agentic Retrieval       │
│                               │
│  Semantic Search Agent        │
│  Document Retrieval           │
│  Relevance / Reranking        │
└───────────────────────────────┘
   ↓
Context Assembly
   ↓
Answer Generation Agent
   ↓
Citation / Grounding Validation
   ↓
Final Response
   ↓
FastAPI
   ↓
Streamlit UI

The exact architecture can change if you identify a better production-oriented design.

---

# MULTI-AGENT DESIGN

Use LangGraph to orchestrate specialized agents/nodes.

At minimum, design the following logical roles:

## 1. Query Understanding / Router Agent

Responsibilities:

- understand the user's question
- determine the type of query
- identify useful retrieval strategy
- determine whether the question requires document retrieval
- rewrite ambiguous queries when necessary
- route the request through the appropriate workflow

Example:

"What authentication mechanism does our API use?"

should become something like:

intent = technical_documentation
retrieval_required = true
search_query = "API authentication mechanism JWT OAuth authentication"

---

## 2. Retrieval Agent

Responsibilities:

- perform semantic retrieval against FAISS
- retrieve relevant chunks
- return source metadata
- avoid blindly returning a large amount of context
- support top-k retrieval
- expose similarity/relevance information

Keep retrieval modular so the vector store can later be replaced with another vector database.

---

## 3. Relevance / Context Agent

Responsibilities:

- inspect retrieved chunks
- determine which chunks are actually relevant
- remove weak/noisy results
- identify missing information
- optionally trigger another retrieval attempt
- assemble a high-quality context package for the answer generator

The system should not assume that the first retrieval is always sufficient.

---

## 4. Answer Generation Agent

Responsibilities:

- answer using retrieved evidence
- follow the system prompt
- never invent unsupported facts
- explicitly say when the knowledge base does not contain enough information
- provide citations to source documents
- distinguish facts from uncertainty

The answer should prioritize retrieved enterprise knowledge over general model knowledge.

---

## 5. Citation / Grounding Validation Agent

Responsibilities:

- inspect the generated answer
- verify that important claims are supported by retrieved context
- verify that citations correspond to actual source chunks
- identify unsupported claims
- request regeneration or correction when necessary

This is important because I want the project to demonstrate **grounded generation**, not simply RAG.

---

# RAG PIPELINE

Implement a proper ingestion pipeline.

## Document ingestion

Support PDF documents initially.

Pipeline:

PDF
 ↓
Text extraction
 ↓
Document normalization
 ↓
Metadata extraction
 ↓
Chunking
 ↓
Embedding generation
 ↓
FAISS indexing
 ↓
Persisted vector index

Each chunk should retain metadata such as:

- document_id
- document_name
- page_number
- chunk_id
- source
- section if available

Design the metadata structure carefully because citations will depend on it.

---

# CHUNKING

Do not use arbitrary fixed-size splitting without thinking about document structure.

Start with a sensible chunking strategy and make it configurable.

Consider:

- chunk size
- chunk overlap
- headings
- paragraphs
- page boundaries
- metadata preservation

Document the reasoning behind the chosen strategy.

---

# EMBEDDINGS

Use a proper embedding model and keep the embedding layer abstracted.

Do not hard-code the embedding implementation throughout the application.

Create a clean interface so the embedding model can later be replaced.

---

# VECTOR STORE

Use **FAISS** for the initial implementation.

Requirements:

- persistent local index
- metadata storage
- document IDs
- chunk IDs
- ability to rebuild the index
- ability to add documents
- ability to inspect index statistics

Do not mix vector-store logic with agent logic.

---

# RETRIEVAL

Implement semantic search.

Start with:

- top-k retrieval
- similarity scores
- metadata filtering where useful

Design the retrieval layer so that future improvements can include:

- hybrid search
- BM25
- reranking
- query expansion
- multi-query retrieval

Do not implement unnecessary complexity immediately.

---

# CITATIONS

Citations are a core feature.

A generated response should be able to reference sources such as:

[Source: engineering_architecture.pdf, page 12]

or another clean citation format.

The citation must map to an actual retrieved document chunk.

Never fabricate citations.

---

# HALLUCINATION CONTROL

The system prompt should enforce rules such as:

1. Answer primarily from retrieved enterprise context.
2. Do not invent enterprise-specific information.
3. If the retrieved context does not contain the answer, explicitly state that.
4. Cite supporting sources.
5. Do not cite a document that does not support the claim.
6. Separate retrieved facts from assumptions.

Implement grounding validation rather than relying only on prompt instructions.

---

# LANGGRAPH

Use LangGraph as the actual orchestration framework.

Define a typed/shared state containing appropriate information such as:

- original_query
- rewritten_query
- intent
- retrieval_required
- retrieved_documents
- filtered_context
- answer
- citations
- grounding_result
- retry_count
- errors

Do not put arbitrary mutable global state into the graph.

Make the graph understandable and observable.

I want to be able to explain the graph in an interview.

---

# FASTAPI

Expose the system through a clean REST API.

At minimum consider endpoints such as:

POST /documents/upload

POST /documents/index

POST /chat

GET /documents

GET /health

The exact endpoint design can be improved if necessary.

The API should:

- validate requests
- handle errors cleanly
- use environment variables for secrets
- return structured responses
- avoid exposing internal implementation details

Example response:

{
  "answer": "...",
  "citations": [
    {
      "document": "architecture.pdf",
      "page": 12,
      "chunk_id": "..."
    }
  ],
  "sources": [...]
}

---

# STREAMLIT

Build a clean Streamlit interface.

The UI should support:

1. Uploading documents
2. Indexing documents
3. Asking questions
4. Viewing answers
5. Viewing citations/sources
6. Seeing retrieved context when useful for debugging/demo purposes

The UI should feel like an enterprise knowledge assistant rather than a generic chatbot.

---

# PROJECT STRUCTURE

Design a clean modular architecture.

A possible structure:

app/
    api/
        routes/
    agents/
    graph/
    rag/
        ingestion/
        chunking/
        embeddings/
        retrieval/
        vectorstore/
    models/
    services/
    prompts/
    config/
    utils/

tests/

scripts/

streamlit/

docs/

The exact structure should be based on the existing repository and refined after inspection.

Do not create unnecessary abstractions.

---

# CONFIGURATION

Use environment variables.

For example:

OPENAI_API_KEY=

Do not hard-code API keys.

Create proper configuration management.

Add:

.env.example

and make sure secrets are excluded through .gitignore.

---

# OBSERVABILITY

Design the system so I can understand what happened during a request.

At minimum I should be able to inspect:

- query
- routing decision
- retrieval query
- retrieved chunks
- relevance decision
- generated answer
- citations
- grounding validation
- retry/fallback decision

Use structured logging where appropriate.

Do not expose secrets.

---

# ERROR HANDLING

Handle realistic failures:

- missing documents
- empty retrieval results
- invalid PDF
- embedding failure
- FAISS loading failure
- OpenAI API failure
- malformed model response
- grounding failure
- duplicate documents
- corrupted index

The system should fail gracefully.

---

# TESTING

Do not wait until the end to test everything.

Build tests incrementally.

At minimum include:

### Unit tests

- chunking
- metadata handling
- retrieval
- citation generation
- configuration
- document processing

### Agent/graph tests

Test:

- routing
- retrieval workflow
- no-result scenario
- grounding failure
- retry behavior

### API tests

Test:

- health endpoint
- document upload
- chat
- invalid requests
- failure scenarios

Create a small evaluation dataset containing realistic enterprise questions and expected source documents.

---

# EVALUATION

Do not judge the RAG system only by whether it "looks good."

Create an evaluation approach.

Measure things such as:

- retrieval relevance
- answer correctness
- citation correctness
- groundedness
- failure-to-answer behavior
- latency
- token usage where practical

Start with a small manually curated evaluation dataset.

Create a script that can run the evaluation repeatedly.

---

# SECURITY

Treat this as an enterprise-oriented project.

At minimum:

- secrets through environment variables
- no API keys in source code
- input validation
- file type validation
- file size limits
- safe file handling
- avoid logging sensitive document contents unnecessarily
- clear separation between user input and system prompts

---

# DEVELOPMENT STRATEGY

IMPORTANT:

Do NOT attempt to build the entire application in one step.

Build it in vertical slices.

Use this progression unless repository inspection suggests a better one:

## Phase 0 — Architecture

Inspect repository and finalize architecture.

## Phase 1 — Foundation

Set up:

- configuration
- dependencies
- project structure
- logging
- basic models
- environment management

Make sure the application runs.

## Phase 2 — Document ingestion

Implement:

PDF → text → chunks → metadata

Test this independently.

## Phase 3 — Embeddings + FAISS

Implement:

chunks → embeddings → FAISS

Add persistence and metadata mapping.

Test retrieval manually.

## Phase 4 — Basic RAG

Implement:

query → retrieval → context → LLM → answer

Before introducing multi-agent complexity, make sure basic RAG works correctly.

## Phase 5 — LangGraph

Convert the workflow into a proper LangGraph state machine.

Implement routing, retrieval, context construction, generation and validation.

## Phase 6 — Multi-agent behavior

Introduce specialized agent responsibilities.

Do not create agents merely for the sake of having multiple agents.

Every agent must have a meaningful responsibility.

## Phase 7 — Grounding + citations

Implement citation tracking and grounding validation.

## Phase 8 — FastAPI

Expose the system through REST APIs.

## Phase 9 — Streamlit

Build the user interface.

## Phase 10 — Testing + evaluation

Add automated tests and RAG evaluation.

## Phase 11 — Production hardening

Improve:

- error handling
- logging
- configuration
- security
- performance
- documentation

## Phase 12 — Documentation

Create a strong README explaining:

- problem statement
- architecture
- system flow
- LangGraph workflow
- RAG pipeline
- multi-agent design
- technologies
- setup
- API usage
- evaluation
- limitations
- future improvements

---

# IMPORTANT CODING BEHAVIOR

For every phase:

1. Explain what you are going to implement.
2. Inspect relevant existing code.
3. Implement the smallest useful increment.
4. Run tests/type checks/linting where available.
5. Manually verify the feature.
6. Fix errors before moving forward.
7. Update documentation.
8. Show me what changed.
9. Explain why the implementation was designed that way.
10. Then recommend the next phase.

Do not silently skip errors.

If something fails, diagnose the root cause instead of applying random patches.

---

# ENGINEERING QUALITY

Write code that I can confidently discuss in an AI Engineer interview.

Avoid:

- giant files
- duplicated logic
- hard-coded configuration
- global mutable state
- unnecessary abstractions
- fake agents
- fake citations
- fake evaluation metrics
- placeholder implementations presented as finished
- excessive framework usage without justification

Prefer:

- clear interfaces
- typed state
- modular components
- dependency injection where useful
- testable functions
- meaningful error handling
- clean configuration
- observable workflows

---

# LEARNING MODE

I am building this project to become a stronger AI Engineer, not just to get a working repository.

Therefore, whenever you introduce an important concept, briefly explain:

- what it does
- why we need it
- why we chose this implementation
- what alternative approaches exist
- what tradeoff we are making

Do not give long theoretical lectures unless I ask.

Teach me through the implementation.

---

# FINAL PROJECT QUALITY BAR

When complete, I should be able to explain this project from end to end:

User question
→ query understanding
→ LangGraph routing
→ retrieval
→ semantic search
→ context filtering
→ answer generation
→ citation generation
→ grounding validation
→ final response
→ FastAPI
→ Streamlit

I should also understand:

- why RAG is needed
- how document ingestion works
- why chunking matters
- how embeddings work
- how FAISS works
- how retrieval works
- why multi-agent architecture is useful
- how LangGraph manages state
- how citations are generated
- how hallucinations are controlled
- how the API exposes the system
- how the system can be evaluated
- how the system could evolve into production

---

# FIRST TASK

Do NOT start implementing yet.

First inspect the repository thoroughly.

Then provide:

### 1. Repository assessment
What exists today.

### 2. Architecture proposal
The architecture you recommend.

### 3. Component responsibilities
What each major component will do.

### 4. LangGraph design
Nodes, state, edges and control flow.

### 5. RAG design
Ingestion, chunking, embeddings, FAISS, retrieval and citations.

### 6. API design
Endpoints and request/response structure.

### 7. Project phases
The incremental implementation roadmap.

### 8. Risks and tradeoffs
Important engineering decisions.

### 9. First implementation step
The smallest useful vertical slice we should build first.

Wait for my approval before beginning major implementation.

Remember: **build a real system incrementally, test everything, and keep the architecture understandable.**