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
    Claim,
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
from rag import confidence, conflicts, generator, llm, retriever, validator
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


def _hits_table(hits: list[ScoredChunk]) -> list[dict]:
    return [{
        "rank": n, "document": h.chunk.filename, "section": " > ".join(h.chunk.section_path) or "(introduction)",
        "cosine": h.dense_score, "dense_rank": h.dense_rank, "bm25_rank": h.bm25_rank, "rrf": h.rrf_score,
    } for n, h in enumerate(hits, start=1)]


def _not_found_answer(hits: list[ScoredChunk]) -> str:
    titles = list(dict.fromkeys(h.chunk.doc_title for h in hits[:3]))
    if not titles:
        return "No documents are indexed yet."
    closest = ", ".join(titles[:-1]) + (" and " if len(titles) > 1 else "") + titles[-1]
    return ("I couldn't find information about this in the uploaded documents. "
            f"The closest documents searched were {closest}, which don't appear to cover it.")


def _extractive(question: str, hits: list[ScoredChunk], candidates: list[conflicts.ConflictCandidate]):
    """Fallback when the LLM is unavailable: quote the top passages with citations."""
    citations = build_citations(hits[:3], question)
    cid_of_hit = {i: c.id for i, c in enumerate(citations)}
    by_id = {c.id: c for c in citations}
    found = [conflicts.candidate_to_conflict(c, cid_of_hit, by_id) for c in candidates
             if c.i in cid_of_hit and c.j in cid_of_hit]
    claims = [Claim(text=c.excerpt, citation_ids=[c.id]) for c in citations]
    answer = "\n".join(f"- {c.excerpt} [{c.id}]" for c in citations)
    return claims, citations, found, answer


def query_documents(question: str, top_k: int = config.TOP_K, filters: Filters | None = None,
                    debug: bool = False) -> QueryResult:
    """Retrieve -> gate -> conflict rules -> one LLM call -> validate -> QueryResult.

    Never raises for LLM problems: falls back to an extractive answer with degraded=True.
    """
    start = time.perf_counter()
    question = question.strip()
    hits = retrieve(question, k=top_k, filters=filters)
    retrieval_ms = int((time.perf_counter() - start) * 1000)
    assessment = confidence.assess(question, hits)
    dbg: dict = {"hits": _hits_table(hits), "llm_called": False,
                 "gate": {"decision": assessment.decision, "top_score": assessment.top_score, "threshold": config.T_LOW}}

    def result(status, answer, gen_ms=0, **kw) -> QueryResult:
        total = int((time.perf_counter() - start) * 1000)
        return QueryResult(
            query_id=f"q_{uuid.uuid4().hex[:6]}", question=question, status=status, answer=answer,
            confidence=kw.pop("conf", assessment.confidence),
            timings_ms={"retrieval": retrieval_ms, "generation": gen_ms, "total": total},
            debug=dbg if debug else None, **kw,
        )

    if assessment.decision == "not_found":
        return result("not_found", _not_found_answer(hits))
    if assessment.decision == "clarify":
        options = assessment.clarify_options
        q = f"Your question could refer to several things: {', '.join(options[:-1])} or {options[-1]}. Which one do you mean?"
        return result("needs_clarification", q, clarifying_question=q, citations=build_citations(hits[:3], question))

    candidates = conflicts.detect(hits)
    dbg["conflict_candidates"] = [c.summary for c in candidates]
    gen_start = time.perf_counter()
    try:
        draft, resp = generator.generate(question, hits, candidates)
    except llm.LLMError as exc:
        dbg["llm_error"] = str(exc)[:300]
        claims, citations, found, answer = _extractive(question, hits, candidates)
        conf = confidence.lower_level(assessment.confidence, "Language model unavailable: showing quoted passages")
        return result("conflict" if found else "answered", answer, int((time.perf_counter() - gen_start) * 1000),
                      claims=claims, citations=citations, conflicts=found, degraded=True, conf=conf)
    gen_ms = int((time.perf_counter() - gen_start) * 1000)
    dbg.update(llm_called=True, llm_cached=resp.cached)

    v = validator.validate(draft, hits)
    dbg["validator"] = {"dropped_claims": v.dropped_claims, "invalid_ids": v.invalid_ids}
    conf = assessment.confidence
    llm_status = draft.get("status")

    if llm_status == "needs_clarification" and draft.get("clarifying_question"):
        q = str(draft["clarifying_question"])
        return result("needs_clarification", q, gen_ms, clarifying_question=q, model=resp.model,
                      citations=build_citations(hits[:3], question), conf=conf)
    if llm_status == "not_found" or not v.claims:
        conf = Confidence(level="low", score=min(conf.score, 0.3),
                          reasons=conf.reasons + ["The model found the retrieved evidence insufficient"])
        return result("not_found", _not_found_answer(hits), gen_ms, model=resp.model, conf=conf)

    by_id = {c.id: c for c in v.citations}
    found = conflicts.merge(v.conflicts, candidates, v.cid_of_hit, by_id, set(v.cid_of_hit))
    if v.dropped_claims and v.dropped_claims * 3 > v.dropped_claims + len(v.claims):
        conf = confidence.lower_level(conf, f"{v.dropped_claims} unsupported claim(s) removed")
    if found:
        conf = conf.model_copy(update={"reasons": conf.reasons + ["Sources disagree"]})
        status = "conflict"
    elif llm_status == "partial" or draft.get("missing_info") or v.dropped_claims * 3 > v.dropped_claims + len(v.claims):
        status = "partial"
    else:
        status = "answered"
    answer = v.answer
    if status == "partial" and draft.get("missing_info"):
        answer += f" Not covered by the documents: {draft['missing_info']}"
    return result(status, answer, gen_ms, claims=v.claims, citations=v.citations, conflicts=found,
                  model=resp.model, conf=conf)


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
            print(query_documents(question, debug="--debug" in argv).model_dump_json(indent=1, exclude_none=True))


if __name__ == "__main__":
    _cli(sys.argv[1:])
