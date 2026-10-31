"""Retrieval: known questions find known passages; persistence; filters."""

import pytest

import config
from common.schemas import Filters
from rag import engine


@pytest.fixture(scope="module", autouse=True)
def corpus_index(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    old = (config.INDEX_DIR, config.UPLOAD_DIR, engine._loaded)
    config.INDEX_DIR, config.UPLOAD_DIR = data / "index", data / "uploads"
    engine._loaded = False
    engine.reset()
    for path in sorted(config.CORPUS_DIR.iterdir()):
        if path.suffix.lower() in config.ALLOWED_EXTENSIONS:
            engine.ingest_file(path)
    yield
    config.INDEX_DIR, config.UPLOAD_DIR, engine._loaded = old
    engine._index.clear()


def _top_files(question, k=3, mode="hybrid"):
    return [h.chunk.filename for h in engine.retrieve(question, k=k, mode=mode)]


@pytest.mark.parametrize("question,expected", [
    ("What does ERR-429 mean?", "troubleshooting.txt"),
    ("My teammate can't get into their account, what should an admin do?", "admin_guide_v3.pdf"),
    ("How do I configure single sign-on?", "admin_guide_v3.pdf"),
    ("How many seats does the Pro plan include?", "usage_limits.html"),
    ("How do I create a workspace?", "onboarding_guide.txt"),
])
def test_known_question_in_top3(question, expected):
    assert expected in _top_files(question)


def test_conflict_question_retrieves_both_sides():
    files = _top_files("What's the refund window for annual plans?")
    assert {"billing_policy_v2.pdf", "billing_faq.md"} <= set(files)


def test_exact_error_code_tops_bm25():
    assert engine.retrieve("ERR-403", k=1, mode="bm25")[0].chunk.text.startswith("ERR-403")


def test_unanswerable_scores_below_answerable():
    gap = engine.retrieve("What is the parental leave policy?", k=1)[0].dense_score
    known = engine.retrieve("What's the API rate limit?", k=1)[0].dense_score
    assert gap < known


def test_filters_and_duplicates_and_persistence():
    doc_id = next(d.doc_id for d in engine.list_documents() if d.filename == "billing_faq.md")
    hits = engine.retrieve("refund", k=5, filters=Filters(doc_ids=[doc_id]))
    assert hits and all(h.chunk.doc_id == doc_id for h in hits)
    status, _ = engine.ingest_file(config.CORPUS_DIR / "billing_faq.md")
    assert status == "duplicate"
    before = engine.index_stats()
    engine._loaded = False
    engine.load()
    assert engine.index_stats() == before
