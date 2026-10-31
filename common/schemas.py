"""Shared data contracts for every module (FROZEN at Gate A).

Owner: ALL, Member 1 is gatekeeper. After the freeze, changes must be additive
(new optional fields only) and agreed by at least two members.

Status precedence used by rag.engine when building a QueryResult:
  1. gate decision not_found / needs_clarification (no LLM call)
  2. conflict   -- at least one conflict survives the merge
  3. partial    -- LLM reported missing_info, or validator dropped > 1/3 of claims
  4. answered
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

DocType = Literal["pdf", "html", "md", "txt"]
DocKind = Literal["manual", "faq", "release_notes", "policy", "support_article", "other"]
Status = Literal["answered", "partial", "conflict", "needs_clarification", "not_found"]
SupportLevel = Literal["supported", "weak"]
ConflictType = Literal["version_sensitive", "value_mismatch", "contradiction"]
DetectedBy = Literal["rules", "llm", "rules+llm"]
ConfidenceLevel = Literal["high", "medium", "low"]
RetrievalMode = Literal["hybrid", "dense", "bm25"]
UploadStatus = Literal["indexed", "duplicate", "failed"]
ErrorCode = Literal[
    "UNSUPPORTED_FORMAT",
    "FILE_TOO_LARGE",
    "EMPTY_DOCUMENT",
    "PARSE_FAILED",
    "INVALID_REQUEST",
    "DOCUMENT_NOT_FOUND",
    "INDEX_EMPTY",
    "INTERNAL_ERROR",
]

DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


# ---------------------------------------------------------------- ingestion


class UserMeta(BaseModel):
    """Optional metadata supplied at upload time; overrides extracted values."""

    version: str | None = None
    effective_date: str | None = Field(default=None, pattern=DATE_PATTERN)
    product: str | None = None
    doc_kind: DocKind | None = None


class DocumentRecord(BaseModel):
    doc_id: str  # first 12 hex chars of sha256(file bytes)
    filename: str
    title: str
    doc_type: DocType
    doc_kind: DocKind = "other"
    version: str | None = None
    effective_date: str | None = Field(default=None, pattern=DATE_PATTERN)
    product: str | None = None
    pages: int | None = None  # PDF only
    num_chunks: int = 0
    uploaded_at: datetime
    source_url: str  # "/documents/{doc_id}/file" (relative to the API base)


class Block(BaseModel):
    """Parser output (ingestion-internal, shared so parsers and chunker agree)."""

    text: str
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None  # 1-based, PDF only


class Chunk(BaseModel):
    chunk_id: str  # "{doc_id}:{ordinal:04d}"
    doc_id: str
    doc_title: str
    filename: str  # copied from DocumentRecord so citations need no registry lookup
    doc_type: DocType
    doc_kind: DocKind = "other"
    product: str | None = None  # copied from DocumentRecord; used by filters + clarify rule
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    ordinal: int
    text: str
    embed_text: str  # "{doc_title} > {section path}\n{text}"
    version: str | None = None
    effective_date: str | None = Field(default=None, pattern=DATE_PATTERN)
    word_count: int


# ---------------------------------------------------------------- retrieval


class Filters(BaseModel):
    doc_ids: list[str] | None = None
    version: str | None = None
    product: str | None = None


class ScoredChunk(BaseModel):
    chunk: Chunk
    dense_score: float  # raw cosine
    dense_rank: int | None = None  # 1-based; None if outside dense top-k
    bm25_rank: int | None = None  # 1-based; None if outside BM25 top-k
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_score: float


class IndexStats(BaseModel):
    documents: int
    chunks: int


# ---------------------------------------------------------------- answers


class Citation(BaseModel):
    id: int = Field(ge=1)  # renumbered 1..n by first use
    chunk_id: str
    doc_id: str
    doc_title: str
    filename: str
    section: str  # section_path joined with " > "
    page: int | None = None
    version: str | None = None
    effective_date: str | None = None
    excerpt: str = Field(max_length=300)
    source_url: str  # relative; UI must prefix API_BASE. PDFs add "#page=N"
    relevance: float


class Claim(BaseModel):
    text: str
    citation_ids: list[int] = Field(min_length=1)
    support: SupportLevel = "supported"


class ConflictPosition(BaseModel):
    citation_id: int
    statement: str
    version: str | None = None
    effective_date: str | None = None


class Conflict(BaseModel):
    type: ConflictType
    topic: str
    summary: str
    positions: list[ConflictPosition] = Field(min_length=2)
    resolution_hint: str | None = None
    detected_by: DetectedBy


class Confidence(BaseModel):
    """score = 0.6*relevance + 0.2*agreement + 0.2*support, where
    relevance = clip((top_dense_cosine - T_LOW) / (1 - T_LOW), 0, 1)."""

    level: ConfidenceLevel
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Filters | None = None
    debug: bool = False


class QueryResult(BaseModel):
    query_id: str
    question: str
    status: Status
    answer: str  # text with [n] citation markers
    claims: list[Claim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    confidence: Confidence
    clarifying_question: str | None = None
    degraded: bool = False
    model: str | None = None
    timings_ms: dict[str, int] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None


# ---------------------------------------------------------------- HTTP


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class UploadItem(BaseModel):
    filename: str
    status: UploadStatus
    document: DocumentRecord | None = None
    error: ErrorBody | None = None


class UploadResponse(BaseModel):
    results: list[UploadItem]
    index: IndexStats


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    index: IndexStats
    llm_provider: str | None = None
