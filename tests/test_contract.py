"""Contract tests: fixture, schemas and engine signatures agree. Owner: Member 2."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import config
from common.schemas import Chunk, Claim, DocumentRecord, QueryResult, ScoredChunk
from rag import engine


def _doc(doc_id: str = "a1b2c3d4e5f6") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        filename="billing_faq.md",
        title="Billing FAQ",
        doc_type="md",
        doc_kind="faq",
        num_chunks=1,
        uploaded_at=datetime.now(timezone.utc),
        source_url=f"/documents/{doc_id}/file",
    )


def _chunk(doc_id: str = "a1b2c3d4e5f6", ordinal: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:{ordinal:04d}",
        doc_id=doc_id,
        doc_title="Billing FAQ",
        filename="billing_faq.md",
        doc_type="md",
        ordinal=ordinal,
        text="Refunds within 14 days.",
        embed_text="Billing FAQ > Refunds\nRefunds within 14 days.",
        word_count=4,
    )


def test_fixture_validates_as_query_result():
    data = json.loads(config.FIXTURE_QUERY_RESULT.read_text(encoding="utf-8"))
    result = QueryResult.model_validate(data)
    assert result.status == "conflict"
    assert {c.id for c in result.citations} >= {i for cl in result.claims for i in cl.citation_ids}


def test_engine_stub_round_trip():
    engine.reset()
    stats = engine.index_chunks(_doc(), [_chunk()])
    assert (stats.documents, stats.chunks) == (1, 1)
    hits = engine.retrieve("refund", k=5, mode="bm25")
    assert hits and isinstance(hits[0], ScoredChunk)
    result = engine.query_documents("What is the refund window?")
    assert isinstance(result, QueryResult)
    assert result.question == "What is the refund window?"
    assert engine.delete_document("a1b2c3d4e5f6").chunks == 0


def test_uncited_claim_is_rejected():
    with pytest.raises(ValidationError):
        Claim(text="Unsupported.", citation_ids=[])


def test_bad_date_is_rejected():
    with pytest.raises(ValidationError):
        DocumentRecord.model_validate({**_doc().model_dump(), "effective_date": "March 2025"})
