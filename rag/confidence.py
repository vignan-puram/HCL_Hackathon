"""Confidence and fallback (FR-07): score the evidence and decide answer / clarify / not_found
before any LLM call is made."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from common.schemas import Confidence, ScoredChunk
from rag.index import tokenize

_GENERIC = set("limit limits policy rule rules setting settings thing it this that one value number much many "
               "long get work use need know tell about reset kept".split())


@dataclass
class Assessment:
    decision: str  # answer | clarify | not_found
    confidence: Confidence
    top_score: float
    clarify_options: list[str] = field(default_factory=list)


def _level(score: float) -> str:
    if score >= config.CONF_HIGH:
        return "high"
    return "medium" if score >= config.CONF_MEDIUM else "low"


def lower_level(conf: Confidence, reason: str) -> Confidence:
    level = {"high": "medium", "medium": "low", "low": "low"}[conf.level]
    return conf.model_copy(update={"level": level, "reasons": conf.reasons + [reason]})


def score(hits: list[ScoredChunk]) -> Confidence:
    top_hit = max(hits, key=lambda h: h.dense_score)
    top = top_hit.dense_score
    relevance = min(max((top - config.T_LOW) / (1 - config.T_LOW), 0.0), 1.0)
    best = hits[0]
    agreement = ((best.dense_rank or 99) <= 5) * 0.5 + ((best.bm25_rank or 99) <= 5) * 0.5
    close = sum(1 for h in hits if h.dense_score >= top - 0.05)
    support = min(close, 3) / 3
    value = round(0.6 * relevance + 0.2 * agreement + 0.2 * support, 3)

    reasons = [f"Best match relevance {top:.2f} (not-found threshold {config.T_LOW:.2f})"]
    if agreement == 1:
        reasons.append("Top passage found by both keyword and semantic search")
    elif agreement == 0.5:
        reasons.append("Top passage found by only one search method")
    docs = len({h.chunk.doc_id for h in hits if h.dense_score >= top - 0.05})
    reasons.append(f"{close} closely matching passage(s) in {docs} document(s)")
    return Confidence(level=_level(value), score=value, reasons=reasons)


def _normalize_section(name: str) -> str:
    """Normalize section titles so '3. Refunds' and 'Refunds' are seen as the same topic."""
    return re.sub(r"^\d+[\.\)]\s*", "", name).strip().lower()


def _is_vague(question: str) -> bool:
    """True when question lacks domain-specific nouns (e.g. 'What is the limit?' or 'How do I reset it?')."""
    q_lower = question.lower()
    has_pronoun = bool(re.search(r"\b(it|this|that)\b", q_lower))
    content = [t for t in tokenize(question) if len(t) > 1]
    specific = [t for t in content if t not in _GENERIC]
    if len(specific) == 0:
        return True
    if has_pronoun and len(specific) <= 1 and len(content) <= 3:
        return True
    return False


def assess(question: str, hits: list[ScoredChunk]) -> Assessment:
    if not hits:
        conf = Confidence(level="low", score=0.0, reasons=["No documents indexed"])
        return Assessment("not_found", conf, 0.0)
    conf = score(hits)
    top = max(h.dense_score for h in hits)
    best = hits[0]

    # Keyword safety: if a query hits relevant content words with solid BM25 and cosine, don't drop it prematurely
    content_words = [w for w in tokenize(question) if w not in _GENERIC and len(w) > 2]
    has_strong_keyword_match = any(
        h.bm25_rank is not None and h.bm25_rank <= 5 and h.dense_score >= 0.50
        and any(w in h.chunk.embed_text.lower() for w in content_words)
        for h in hits[:3]
    )
    if top < config.T_LOW and not has_strong_keyword_match:
        conf = Confidence(level="low", score=min(conf.score, 0.2),
                          reasons=[f"Best match relevance {top:.2f} is below the threshold {config.T_LOW:.2f}"])
        return Assessment("not_found", conf, top)

    # vague question whose close matches are spread over distinctly different topics -> ask which one
    close = [h for h in hits if h.dense_score >= top - config.CLARIFY_MARGIN * 3]
    seen_norms = set()
    distinct_sections = []
    for h in close:
        sec = h.chunk.section_path[-1] if h.chunk.section_path else h.chunk.doc_title
        norm = _normalize_section(sec)
        if norm and norm not in seen_norms:
            seen_norms.add(norm)
            distinct_sections.append(sec)

    if _is_vague(question) and len(distinct_sections) >= 2:
        conf = conf.model_copy(update={"reasons": conf.reasons + ["Question is underspecified and matches several topics"]})
        return Assessment("clarify", conf, top, clarify_options=distinct_sections[:4])
    return Assessment("answer", conf, top)
