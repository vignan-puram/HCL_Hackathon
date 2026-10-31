"""Answer pipeline without network: validator, conflict rules, gate decisions, LLM fallback."""

import pytest

import config
from common.schemas import Chunk, ScoredChunk
from rag import confidence, conflicts, engine, llm
from rag.validator import validate


def _hit(doc_id, text, score=0.8, version=None, date=None, title="Doc"):
    chunk = Chunk(chunk_id=f"{doc_id}:0000", doc_id=doc_id, doc_title=title, filename=f"{doc_id}.md", doc_type="md",
                  section_path=["Refunds"], ordinal=0, text=text, embed_text=f"{title} > Refunds\n{text}",
                  version=version, effective_date=date, word_count=len(text.split()))
    return ScoredChunk(chunk=chunk, dense_score=score, dense_rank=1, bm25_rank=1, rrf_score=0.03, final_score=0.03)


HITS = [
    _hit("policy000001", "Annual subscriptions may be refunded within 30 days of purchase.", version="2.0",
         date="2025-03-01", title="Billing Policy"),
    _hit("faq000000001", "You can request a refund within 14 days of your annual renewal.", score=0.78,
         date="2023-06-10", title="Billing FAQ"),
]


def test_validator_drops_invented_ids_and_uncited_claims():
    draft = {"status": "answered", "claims": [
        {"text": "Annual plans can be refunded within 30 days.", "evidence": ["S1"]},
        {"text": "Refunds take one hour.", "evidence": ["S9"]},
        {"text": "No citation here.", "evidence": []},
    ]}
    v = validate(draft, HITS)
    assert len(v.claims) == 1 and v.dropped_claims == 2
    assert v.invalid_ids == ["S9"]
    assert [c.id for c in v.citations] == [1] and v.citations[0].doc_id == "policy000001"
    assert v.answer == "Annual plans can be refunded within 30 days [1]."


def test_validator_renumbers_by_first_use_and_flags_weak_support():
    draft = {"claims": [{"text": "The FAQ allows a refund within 14 days.", "evidence": ["S2"]},
                        {"text": "The policy allows 45 days.", "evidence": ["S1"]}]}
    v = validate(draft, HITS)
    assert v.citations[0].doc_id == "faq000000001" and v.claims[1].citation_ids == [2]
    assert v.claims[0].support == "supported" and v.claims[1].support == "weak"  # 45 is not in the source


def test_value_conflict_rule_and_scoped_values():
    found = conflicts.detect(HITS)
    assert len(found) == 1 and found[0].unit == "days"
    scoped = [_hit("ret000000001", "Enterprise: deleted data is retained for 180 days. Free and Pro: deleted data is retained for 90 days."),
              _hit("sec000000001", "Deleted data is retained for 90 days before it is permanently erased.", score=0.79)]
    assert conflicts.detect(scoped) == []


def test_gate_refuses_weak_evidence_and_clarifies_vague_questions():
    weak = [_hit("x00000000001", "Unrelated text.", score=config.T_LOW - 0.05)]
    assert confidence.assess("What is the parental leave policy?", weak).decision == "not_found"
    spread = [_hit("a00000000001", "Users per plan.", score=0.70), _hit("b00000000001", "Storage per plan.", score=0.69)]
    spread[1].chunk.section_path = ["Storage"]
    assert confidence.assess("What's the limit?", spread).decision == "clarify"
    assert confidence.assess("What is the refund window for annual plans?", HITS).decision == "answer"


def test_llm_failure_returns_degraded_extractive_answer(monkeypatch):
    monkeypatch.setattr(engine, "retrieve", lambda *a, **k: HITS)

    def boom(*a, **k):
        raise llm.LLMError("quota exhausted")

    monkeypatch.setattr(llm, "complete_json", boom)
    result = engine.query_documents("What is the refund window for annual plans?", debug=True)
    assert result.degraded and result.citations and result.status == "conflict"
    assert "quota exhausted" in result.debug["llm_error"]


def test_llm_answer_is_validated_end_to_end(monkeypatch):
    monkeypatch.setattr(engine, "retrieve", lambda *a, **k: HITS)
    reply = ('{"status": "conflict", "claims": [{"text": "The policy allows 30 days.", "evidence": ["S1"]}, '
             '{"text": "The FAQ allows 14 days.", "evidence": ["S2"]}], "conflicts": []}')
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: llm.LLMResponse(reply, "fake-model", False))
    result = engine.query_documents("What is the refund window for annual plans?")
    assert result.status == "conflict" and result.model == "fake-model"
    assert result.conflicts[0].detected_by == "rules"  # rule kept although the LLM ignored it
    assert {p.statement for p in result.conflicts[0].positions} == {"30 days", "14 days"}


def test_conflict_only_draft_becomes_cited_claims():
    draft = {"status": "conflict", "claims": [], "conflicts": [{
        "type": "value_mismatch", "topic": "Refund window", "summary": "30 vs 14 days",
        "positions": [{"evidence": "S1", "statement": "Refunds within 30 days."},
                      {"evidence": "S2", "statement": "Refunds within 14 days."}]}]}
    v = validate(draft, HITS)
    assert [c.citation_ids for c in v.claims] == [[1], [2]]
    assert v.claims[0].text.startswith("Billing Policy v2.0 (2025-03-01)")
    assert "[1]" in v.answer and "[2]" in v.answer and len(v.conflicts) == 1


def test_conversational_and_acronym_responses():
    engine.load()
    greet = engine.query_documents("Hello")
    assert greet.status == "answered"
    assert "Enterprise Documentation Assistant" in greet.answer

    help_res = engine.query_documents("What can you do?")
    assert help_res.status == "answered"
    assert "Billing & Refunds" in help_res.answer

    acronym_res = engine.query_documents("what does SSO stand for?")
    assert acronym_res.status == "answered"
    assert "Single Sign-On" in acronym_res.answer
    assert len(acronym_res.citations) >= 1
