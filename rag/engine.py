"""PUBLIC engine API -- the only rag module the app and the eval import.

CLI (from repo root):
  python -m rag.engine index-corpus          # (re)index eval/corpus
  python -m rag.engine ask "question" [--mode dense|bm25|hybrid] [--retrieve]
"""

from __future__ import annotations

import shutil
import sys
import time
import uuid
from pathlib import Path

import config
from common.schemas import (
    Chunk,
    Confidence,
    DocumentRecord,
    Filters,
    IndexStats,
    QueryResult,
    RetrievalMode,
    ScoredChunk,
    UserMeta,
)
from ingestion.errors import IngestionError
from ingestion.pipeline import doc_id_for, process_file
from rag import retriever
from rag.embeddings import encode
from rag.index import Index
from rag.validator import build_citations

_index = Index()
_loaded = False


def load() -> None:
    """Load the persisted index once per process."""
    global _loaded
    if not _loaded:
        _index.load(config.INDEX_DIR)
        _loaded = True


def _persist() -> None:
    _index.save(config.INDEX_DIR)


# ---------------------------------------------------------------- documents


def index_chunks(doc: DocumentRecord, chunks: list[Chunk]) -> IndexStats:
    load()
    vectors = encode([c.embed_text for c in chunks])
    _index.add(doc, chunks, vectors)
    _persist()
    return index_stats()


def ingest_file(path: Path, user_meta: UserMeta | None = None,
                filename: str | None = None) -> tuple[str, DocumentRecord]:
    """Ingest one file. Returns (status, record) with status 'indexed' or 'duplicate'.

    Raises IngestionError for unsupported, empty or unparseable files.
    """
    load()
    path = Path(path)
    filename = filename or path.name
    if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise IngestionError("UNSUPPORTED_FORMAT", f"'{path.suffix}' is not supported. Use PDF, HTML, Markdown or TXT.")
    doc_id = doc_id_for(path.read_bytes())
    if doc_id in _index.docs and not user_meta:
        return "duplicate", _index.docs[doc_id]
    record, chunks = process_file(path, user_meta, filename=filename)
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored = config.UPLOAD_DIR / f"{record.doc_id}_{filename}"
    if path.resolve() != stored.resolve():
        shutil.copyfile(path, stored)
    index_chunks(record, chunks)
    return "indexed", record


def delete_document(doc_id: str) -> IndexStats:
    load()
    _index.delete(doc_id)
    for f in config.UPLOAD_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)
    _persist()
    return index_stats()


def reset() -> IndexStats:
    load()
    _index.clear()
    shutil.rmtree(config.INDEX_DIR, ignore_errors=True)
    shutil.rmtree(config.UPLOAD_DIR, ignore_errors=True)
    return index_stats()


def list_documents() -> list[DocumentRecord]:
    load()
    return sorted(_index.docs.values(), key=lambda d: d.filename)


def document_chunks(doc_id: str) -> list[Chunk]:
    load()
    return [c for c in _index.chunks if c.doc_id == doc_id]


def original_file(doc_id: str) -> Path | None:
    matches = list(config.UPLOAD_DIR.glob(f"{doc_id}_*"))
    return matches[0] if matches else None


def index_stats() -> IndexStats:
    load()
    return IndexStats(documents=len(_index.docs), chunks=len(_index.chunks))


# ---------------------------------------------------------------- questions


def retrieve(question: str, k: int = config.TOP_K, filters: Filters | None = None,
             mode: RetrievalMode = "hybrid") -> list[ScoredChunk]:
    load()
    return retriever.retrieve(_index, question, k=k, filters=filters, mode=mode)


def query_documents(question: str, top_k: int = config.TOP_K, filters: Filters | None = None,
                    debug: bool = False) -> QueryResult:
    """Extractive mode (Milestone 4): top passages as cited evidence, no LLM."""
    start = time.perf_counter()
    hits = retrieve(question, k=top_k, filters=filters)
    retrieval_ms = int((time.perf_counter() - start) * 1000)
    citations = build_citations(hits, question)
    answer = "\n".join(f"- {c.excerpt} [{c.id}]" for c in citations[:3]) or "No documents indexed."
    return QueryResult(
        query_id=f"q_{uuid.uuid4().hex[:6]}",
        question=question,
        status="answered" if citations else "not_found",
        answer=answer,
        citations=citations,
        confidence=Confidence(level="low", score=0.0, reasons=["Extractive mode: no LLM answer generated"]),
        degraded=True,
        timings_ms={"retrieval": retrieval_ms, "generation": 0, "total": retrieval_ms},
    )


# ---------------------------------------------------------------- CLI


def _cli(argv: list[str]) -> None:
    if not argv or argv[0] == "index-corpus":
        reset()
        for path in sorted(config.CORPUS_DIR.iterdir()):
            if path.suffix.lower() in config.ALLOWED_EXTENSIONS:
                status, rec = ingest_file(path)
                print(f"{status:9} {rec.filename:28} chunks={rec.num_chunks}")
        print(index_stats())
        return
    if argv[0] == "ask":
        question = argv[1]
        mode = argv[argv.index("--mode") + 1] if "--mode" in argv else "hybrid"
        if "--retrieve" in argv:
            for h in retrieve(question, k=5, mode=mode):
                print(f"{h.final_score:.4f} cos={h.dense_score:.3f} d#{h.dense_rank} b#{h.bm25_rank} "
                      f"{h.chunk.filename} > {' > '.join(h.chunk.section_path)}")
        else:
            print(query_documents(question).model_dump_json(indent=1))


if __name__ == "__main__":
    _cli(sys.argv[1:])
