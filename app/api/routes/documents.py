"""Document management routes."""

import hashlib
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config.settings import settings
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.ingestion import Chunker, ChunkingConfig, PDFProcessor
from app.rag.vectorstore import FAISSVectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    page_count: int | None = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    file_size_bytes: int
    upload_timestamp: str
    indexed: bool = False
    chunk_count: int | None = None


class IndexRequest(BaseModel):
    document_ids: list[str] | None = None


class IndexResponse(BaseModel):
    indexed_count: int
    total_chunks: int
    status: str


_UPLOAD_FILE = File(...)


def _get_upload_path(filename: str) -> Path:
    """Get the path for an uploaded file."""
    upload_dir = Path(settings.data_uploads_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / filename


def _get_index_dir() -> Path:
    """Get the index directory."""
    index_dir = Path(settings.data_index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def _get_uploaded_documents() -> list[Path]:
    """Get list of uploaded PDF files."""
    upload_dir = Path(settings.data_uploads_dir)
    if not upload_dir.exists():
        return []
    return list(upload_dir.glob("*.pdf"))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = _UPLOAD_FILE):
    """Upload a PDF document."""
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Read content
    content = await file.read()

    # Check file size
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Compute hash for document ID
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    document_id = f"doc_{file_hash}"

    # Save to uploads directory
    upload_path = _get_upload_path(file.filename)
    upload_path.write_bytes(content)

    # Extract page count
    try:
        import pymupdf
        doc = pymupdf.open(upload_path)
        page_count = doc.page_count
        doc.close()
    except Exception as e:
        logger.warning("Could not read page count", extra={"error": str(e)})
        page_count = None

    logger.info("Document uploaded", extra={
        "document_id": document_id,
        "file_name": file.filename,
        "page_count": page_count,
    })

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="uploaded",
        page_count=page_count,
    )


@router.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Index uploaded documents."""
    index_dir = _get_index_dir()

    # Determine which documents to index
    if request.document_ids:
        # Filter uploaded documents by requested IDs
        all_pdfs = _get_uploaded_documents()
        pdf_paths = [p for p in all_pdfs if p.stem.startswith("doc_")]
        # For simplicity, index all if IDs don't match exactly
    else:
        pdf_paths = _get_uploaded_documents()

    if not pdf_paths:
        raise HTTPException(status_code=404, detail="No PDF documents found to index")

    # Initialize components
    processor = PDFProcessor()
    chunker = Chunker(ChunkingConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ))
    embedder = SentenceTransformerEmbeddingProvider()
    vector_store = FAISSVectorStore(dimension=embedder.dimension)

    total_chunks = 0
    indexed_count = 0

    for pdf_path in pdf_paths:
        try:
            logger.info("Processing document for indexing", extra={"path": str(pdf_path)})

            # Extract document
            document = processor.extract_document(pdf_path)

            # Chunk document
            chunks = chunker.chunk_document(document)

            if not chunks:
                logger.warning("No chunks generated", extra={"path": str(pdf_path)})
                continue

            # Generate embeddings
            texts = [chunk.text for chunk in chunks]
            logger.info("Generating embeddings", extra={"count": len(texts)})
            embeddings = embedder.embed_and_normalize(texts)

            # Add to vector store
            vector_store.add_documents(chunks, embeddings)

            total_chunks += len(chunks)
            indexed_count += 1

        except Exception as e:
            logger.error("Failed to index document", extra={
                "path": str(pdf_path),
                "error": str(e),
            })
            # Continue with other documents

    # Save index
    logger.info("Saving index", extra={"path": str(index_dir)})
    vector_store.save(str(index_dir))

    logger.info("Indexing complete", extra={
        "indexed_count": indexed_count,
        "total_chunks": total_chunks,
        "status": "completed",
    })

    return IndexResponse(
        indexed_count=indexed_count,
        total_chunks=total_chunks,
        status="completed",
    )


@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    """List all uploaded documents."""
    pdf_paths = _get_uploaded_documents()

    documents = []
    for pdf_path in pdf_paths:
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            page_count = doc.page_count
            doc.close()

            stat = pdf_path.stat()

            # Try to extract document ID from filename
            doc_id = f"doc_{hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]}"

            documents.append(DocumentInfo(
                document_id=doc_id,
                filename=pdf_path.name,
                page_count=page_count,
                file_size_bytes=stat.st_size,
                upload_timestamp=str(stat.st_mtime),
                indexed=(Path(settings.data_index_dir) / "index.faiss").exists(),
                chunk_count=None,  # Could be computed if needed
            ))
        except Exception as e:
            logger.warning("Failed to read document info", extra={
                "path": str(pdf_path),
                "error": str(e),
            })

    return documents


@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get document details."""
    pdf_paths = _get_uploaded_documents()

    for pdf_path in pdf_paths:
        try:
            computed_id = f"doc_{hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]}"
            if computed_id == document_id:
                import pymupdf
                doc = pymupdf.open(pdf_path)
                page_count = doc.page_count
                doc.close()

                stat = pdf_path.stat()

                return DocumentInfo(
                    document_id=document_id,
                    filename=pdf_path.name,
                    page_count=page_count,
                    file_size_bytes=stat.st_size,
                    upload_timestamp=str(stat.st_mtime),
                    indexed=(Path(settings.data_index_dir) / "index.faiss").exists(),
                    chunk_count=None,
                )
        except Exception:
            continue

    raise HTTPException(status_code=404, detail="Document not found")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the uploads."""
    pdf_paths = _get_uploaded_documents()

    for pdf_path in pdf_paths:
        try:
            computed_id = f"doc_{hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]}"
            if computed_id == document_id:
                pdf_path.unlink()
                logger.info("Document deleted", extra={"document_id": document_id})
                return {"status": "deleted", "document_id": document_id}
        except Exception:
            continue

    raise HTTPException(status_code=404, detail="Document not found")
