"""FastAPI backend application for Enterprise Documentation Q&A Assistant.

Exposes REST API endpoints defined in the Hackathon Engineering Blueprint:
- GET /health
- POST /query
- POST /upload
- GET /documents
- GET /documents/{doc_id}/file
- GET /documents/{doc_id}/chunks
- DELETE /documents/{doc_id}
- POST /reset
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

import config
from backend.errors import APIError, api_error_handler
from common.schemas import (
    Chunk,
    DocKind,
    DocumentRecord,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    IndexStats,
    QueryRequest,
    QueryResult,
    UploadDocSummary,
    UploadItem,
    UploadItemError,
    UploadResponse,
    UserMeta,
)
from ingestion.errors import IngestionError
import rag.engine as engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load engine index and embeddings at startup."""
    engine.load()
    yield


app = FastAPI(
    title="Enterprise Documentation Q&A Assistant",
    description="Grounded documentation Q&A API with claim-level citations and conflict detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers
app.add_exception_handler(APIError, api_error_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    content = ErrorResponse(
        error=ErrorBody(
            code="INVALID_REQUEST",
            message="Invalid request format or parameters.",
            details={"errors": exc.errors()},
        )
    ).model_dump()
    return JSONResponse(status_code=422, content=content)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    content = ErrorResponse(
        error=ErrorBody(
            code="INTERNAL_ERROR",
            message=str(exc) or "Internal server error.",
            details={},
        )
    ).model_dump()
    return JSONResponse(status_code=500, content=content)


# ---------------------------------------------------------------- Routes


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check, current index size, and active LLM provider."""
    return HealthResponse(
        status="ok",
        index=engine.index_stats(),
        llm_provider=config.LLM_PROVIDER,
    )


@app.post("/query", response_model=QueryResult)
def query(req: QueryRequest) -> QueryResult:
    """Ask a question against indexed enterprise documentation."""
    stats = engine.index_stats()
    if stats.documents == 0:
        raise APIError(
            code="INDEX_EMPTY",
            message="Upload at least one document before asking questions.",
        )

    # Engine query_documents handles answering, gate decisions (not_found, clarify), and extractive fallback
    return engine.query_documents(
        question=req.question,
        top_k=req.top_k,
        filters=req.filters,
        debug=req.debug,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    version: str | None = Form(default=None),
    effective_date: str | None = Form(default=None),
    product: str | None = Form(default=None),
    doc_kind: DocKind | None = Form(default=None),
) -> UploadResponse:
    """Upload one or more documents (PDF, HTML, MD, TXT) with optional metadata overrides."""
    user_meta: UserMeta | None = None
    if any([version, effective_date, product, doc_kind]):
        try:
            user_meta = UserMeta(
                version=version or None,
                effective_date=effective_date or None,
                product=product or None,
                doc_kind=doc_kind or None,
            )
        except ValidationError as e:
            raise APIError("INVALID_REQUEST", f"Invalid metadata provided: {e}")

    results: list[UploadItem] = []
    max_bytes = config.MAX_FILE_MB * 1024 * 1024

    for file in files:
        filename = file.filename or "uploaded_file"
        try:
            content = await file.read()
            if len(content) > max_bytes:
                results.append(
                    UploadItem(
                        filename=filename,
                        status="failed",
                        error=UploadItemError(
                            code="FILE_TOO_LARGE",
                            message=f"File exceeds maximum size of {config.MAX_FILE_MB}MB.",
                        ),
                    )
                )
                continue

            # Write to a temporary file for ingestion pipeline processing
            suffix = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            try:
                status, record = engine.ingest_file(
                    path=tmp_path,
                    user_meta=user_meta,
                    filename=filename,
                )
                results.append(
                    UploadItem(
                        filename=filename,
                        status=status,  # "indexed" or "duplicate"
                        document=UploadDocSummary(
                            doc_id=record.doc_id,
                            title=record.title,
                            num_chunks=record.num_chunks,
                        ),
                    )
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        except IngestionError as e:
            results.append(
                UploadItem(
                    filename=filename,
                    status="failed",
                    error=UploadItemError(code=e.code, message=e.message),
                )
            )
        except Exception as e:
            results.append(
                UploadItem(
                    filename=filename,
                    status="failed",
                    error=UploadItemError(code="PARSE_FAILED", message=str(e)),
                )
            )

    return UploadResponse(
        results=results,
        index=engine.index_stats(),
    )


@app.get("/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    """List all indexed documents with extracted metadata and chunk counts."""
    return engine.list_documents()


@app.get("/documents/{doc_id}/file")
def get_original_file(doc_id: str):
    """Serve the original document file (target for citation links)."""
    file_path = engine.original_file(doc_id)
    if not file_path or not file_path.exists():
        raise APIError("DOCUMENT_NOT_FOUND", f"Document '{doc_id}' not found.")

    media_types = {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".htm": "text/html",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".txt": "text/plain",
    }
    media_type = media_types.get(file_path.suffix.lower(), "application/octet-stream")
    return FileResponse(file_path, media_type=media_type, filename=file_path.name)


@app.get("/documents/{doc_id}/chunks", response_model=list[Chunk])
def get_document_chunks(doc_id: str) -> list[Chunk]:
    """Inspect how a document was cleaned and chunked."""
    chunks = engine.document_chunks(doc_id)
    if not chunks:
        # Check if doc exists at all
        docs = {d.doc_id: d for d in engine.list_documents()}
        if doc_id not in docs:
            raise APIError("DOCUMENT_NOT_FOUND", f"Document '{doc_id}' not found.")
    return chunks


@app.delete("/documents/{doc_id}", response_model=IndexStats)
def delete_document(doc_id: str) -> IndexStats:
    """Remove a document from registry, chunk index, and embeddings."""
    docs = {d.doc_id: d for d in engine.list_documents()}
    if doc_id not in docs:
        raise APIError("DOCUMENT_NOT_FOUND", f"Document '{doc_id}' not found.")
    return engine.delete_document(doc_id)


@app.post("/reset", response_model=IndexStats)
def reset_index() -> IndexStats:
    """Clear all documents and chunks from the knowledge base."""
    return engine.reset()
