"""Citations (FR-05): build Citation objects and pick the supporting excerpt."""

from __future__ import annotations

import re

from common.schemas import Citation, ScoredChunk
from rag.index import tokenize

MAX_EXCERPT = 300


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def pick_excerpt(text: str, focus: str) -> str:
    """The sentence (plus its neighbour, if it fits) that best overlaps the focus text."""
    sentences = _sentences(text) or [text]
    focus_tokens = set(tokenize(focus))
    scores = [len(focus_tokens & set(tokenize(s))) for s in sentences]
    best = max(range(len(sentences)), key=lambda i: (scores[i], -i))
    excerpt = sentences[best]
    if best + 1 < len(sentences) and len(excerpt) + len(sentences[best + 1]) + 1 <= MAX_EXCERPT and scores[best + 1] > 0:
        excerpt += " " + sentences[best + 1]
    if len(excerpt) > MAX_EXCERPT:
        excerpt = excerpt[: MAX_EXCERPT - 1].rsplit(" ", 1)[0] + "…"
    return excerpt


def make_citation(cid: int, hit: ScoredChunk, focus: str) -> Citation:
    c = hit.chunk
    page = c.page_start
    return Citation(
        id=cid,
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        doc_title=c.doc_title,
        filename=c.filename,
        section=" > ".join(c.section_path) or "(introduction)",
        page=page,
        version=c.version,
        effective_date=c.effective_date,
        excerpt=pick_excerpt(c.text, focus),
        source_url=f"{c.filename}#page={page}" if page else c.filename,
        relevance=round(hit.dense_score, 3),
    )


def build_citations(hits: list[ScoredChunk], focus: str) -> list[Citation]:
    return [make_citation(i, h, focus) for i, h in enumerate(hits, start=1)]
