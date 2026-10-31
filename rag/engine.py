"""PUBLIC engine API -- the only rag module imported outside rag/. Owner: Member 1.

STUB (Gate A): keeps chunks in memory, returns them unranked from retrieve(),
and answers every question with the fixture QueryResult. Signatures are final;
the bodies get replaced by the real pipeline without changing callers.
"""

from __future__ import annotations

import json
import uuid

import config
from common.schemas import (
    Chunk,
    DocumentRecord,
    Filters,
    IndexStats,
    QueryResult,
    RetrievalMode,
    ScoredChunk,
)

_docs: dict[str, DocumentRecord] = {}
_chunks: list[Chunk] = []


def load() -> None:
    """Load the persisted index from config.INDEX_DIR. Call once at startup."""


def index_chunks(doc: DocumentRecord, chunks: list[Chunk]) -> IndexStats:
    delete_document(doc.doc_id)
    _docs[doc.doc_id] = doc
    _chunks.extend(chunks)
    return index_stats()


def delete_document(doc_id: str) -> IndexStats:
    global _chunks
    _docs.pop(doc_id, None)
    _chunks = [c for c in _chunks if c.doc_id != doc_id]
    return index_stats()


def retrieve(
    question: str,
    k: int = config.TOP_K,
    filters: Filters | None = None,
    mode: RetrievalMode = "hybrid",
) -> list[ScoredChunk]:
    hits = _chunks
    if filters and filters.doc_ids:
        hits = [c for c in hits if c.doc_id in filters.doc_ids]
    return [ScoredChunk(chunk=c, dense_score=0.0, final_score=0.0) for c in hits[:k]]


def query_documents(
    question: str,
    top_k: int = config.TOP_K,
    filters: Filters | None = None,
    debug: bool = False,
) -> QueryResult:
    data = json.loads(config.FIXTURE_QUERY_RESULT.read_text(encoding="utf-8"))
    data.update(query_id=f"q_{uuid.uuid4().hex[:6]}", question=question)
    return QueryResult.model_validate(data)


def index_stats() -> IndexStats:
    return IndexStats(documents=len(_docs), chunks=len(_chunks))


def reset() -> IndexStats:
    _docs.clear()
    _chunks.clear()
    return index_stats()
