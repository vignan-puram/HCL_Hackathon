"""Conflict detection (FR-06): rules propose, the LLM confirms, nothing is silently dropped.

Rule: two retrieved passages from different documents that are about the same
thing (chunk cosine >= CONFLICT_SIM, both relevant to the question) state the
same unit with disjoint values ("30 days" vs "14 days", "100 requests per
minute" vs "300 requests per minute"). If both documents carry different
version numbers the conflict is version_sensitive, otherwise value_mismatch.
Requiring *disjoint* values keeps scoped statements ("180 days for Enterprise,
90 days for Free and Pro" vs "90 days") from being flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from common.schemas import Citation, Conflict, ConflictPosition, ScoredChunk
from rag.embeddings import encode
from rag.index import tokenize

_UNITS = [
    (r"requests?\s*(?:per|/)\s*min(?:ute)?", "requests per minute"),
    (r"business\s+days?", "business days"),
    (r"days?", "days"),
    (r"hours?", "hours"),
    (r"minutes?", "minutes"),
    (r"months?", "months"),
    (r"years?", "years"),
    (r"GB", "GB"),
    (r"TB", "TB"),
    (r"MB", "MB"),
    (r"users?", "users"),
    (r"%", "%"),
]
_QTY = re.compile(r"(?<![\w.$-])(\d+(?:\.\d+)?)\s*(" + "|".join(p for p, _ in _UNITS) + r")\b", re.I)
_PRICE = re.compile(r"\$(\d+(?:\.\d+)?)")
_RELEVANCE_MARGIN = 0.12
_TOPIC_STOP = set("days hours minutes months years users within after before more than each every until "
                  "will been have from your with this that there which".split())  # both passages must score within this of the best hit


@dataclass
class ConflictCandidate:
    i: int  # hit index (0-based)
    j: int
    type: str
    unit: str
    values_i: list[str]
    values_j: list[str]
    summary: str


def quantities(text: str) -> dict[str, dict[str, str]]:
    """unit -> {value: sentence that states it}."""
    found: dict[str, dict[str, str]] = {}
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        for value, unit in _QTY.findall(sentence):
            name = next(n for p, n in _UNITS if re.fullmatch(p, unit, re.I))
            found.setdefault(name, {}).setdefault(value, sentence)
        for value in _PRICE.findall(sentence):
            found.setdefault("USD", {}).setdefault(value, sentence)
    return found


def _stems(sentences: list[str]) -> set[str]:
    words = (w for s in sentences for w in tokenize(s) if w.isalpha() and len(w) > 3)
    return {w[:5] for w in words if w not in _TOPIC_STOP}


def _same_topic(a: dict[str, str], b: dict[str, str]) -> bool:
    """The sentences stating the two values must share a content word (prefix-stemmed)."""
    return bool(_stems(list(a.values())) & _stems(list(b.values())))


def detect(hits: list[ScoredChunk]) -> list[ConflictCandidate]:
    if len(hits) < 2:
        return []
    top = max(h.dense_score for h in hits)
    relevant = [k for k, h in enumerate(hits) if h.dense_score >= top - _RELEVANCE_MARGIN]
    vectors = encode([hits[k].chunk.embed_text for k in relevant])
    sims = vectors @ vectors.T
    qty = {k: quantities(hits[k].chunk.text) for k in relevant}

    candidates: list[ConflictCandidate] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for a_pos, a in enumerate(relevant):
        for b_pos in range(a_pos + 1, len(relevant)):
            b = relevant[b_pos]
            ca, cb = hits[a].chunk, hits[b].chunk
            if ca.doc_id == cb.doc_id or float(sims[a_pos, b_pos]) < config.CONFLICT_SIM:
                continue
            for unit in sorted(set(qty[a]) & set(qty[b])):
                va, vb = set(qty[a][unit]), set(qty[b][unit])
                key = (min(ca.doc_id, cb.doc_id), max(ca.doc_id, cb.doc_id), unit)
                if va & vb or key in seen_pairs or not _same_topic(qty[a][unit], qty[b][unit]):
                    continue
                seen_pairs.add(key)
                kind = "version_sensitive" if ca.version and cb.version and ca.version != cb.version else "value_mismatch"
                fmt = lambda vals: "/".join(sorted(vals, key=float))  # noqa: E731
                candidates.append(ConflictCandidate(
                    i=a, j=b, type=kind, unit=unit, values_i=sorted(va), values_j=sorted(vb),
                    summary=f"{fmt(va)} {unit} in {ca.doc_title} vs {fmt(vb)} {unit} in {cb.doc_title}",
                ))
    return candidates


def _newer_hint(positions: list[ConflictPosition], by_id: dict[int, Citation]) -> str | None:
    dated = [(p.effective_date, p) for p in positions if p.effective_date]
    if len({d for d, _ in dated}) < 2:
        return None
    newest = max(dated, key=lambda x: x[0])[1]
    cite = by_id.get(newest.citation_id)
    name = cite.doc_title if cite else "The newer source"
    version = f" v{newest.version}" if newest.version else ""
    return f"{name}{version} ({newest.effective_date}) is the most recent source; confirm which applies to your account or version."


def candidate_to_conflict(cand: ConflictCandidate, cid_of_hit: dict[int, int], by_id: dict[int, Citation],
                          detected_by: str = "rules") -> Conflict:
    positions = []
    for hit_index, values in ((cand.i, cand.values_i), (cand.j, cand.values_j)):
        cite = by_id[cid_of_hit[hit_index]]
        positions.append(ConflictPosition(
            citation_id=cite.id, statement=f"{'/'.join(values)} {cand.unit}",
            version=cite.version, effective_date=cite.effective_date,
        ))
    return Conflict(
        type=cand.type, topic=f"Different values for '{cand.unit}'", summary=cand.summary,
        positions=positions, resolution_hint=_newer_hint(positions, by_id), detected_by=detected_by,
    )


def merge(llm_conflicts: list[Conflict], candidates: list[ConflictCandidate], cid_of_hit: dict[int, int],
          by_id: dict[int, Citation], cited_hits: set[int]) -> list[Conflict]:
    """Keep LLM conflicts; mark those a rule also found as rules+llm; keep rule-only conflicts
    the LLM ignored when the answer cites both sides (labelled "possible conflict" in the UI)."""
    merged: list[Conflict] = []
    covered: set[int] = set()
    for conflict in llm_conflicts:
        cited = {p.citation_id for p in conflict.positions}
        for n, cand in enumerate(candidates):
            if {cid_of_hit.get(cand.i), cid_of_hit.get(cand.j)} <= cited:
                covered.add(n)
                conflict = conflict.model_copy(update={"detected_by": "rules+llm"})
        if not conflict.resolution_hint:
            conflict = conflict.model_copy(update={"resolution_hint": _newer_hint(conflict.positions, by_id)})
        merged.append(conflict)
    for n, cand in enumerate(candidates):
        if n not in covered and cand.i in cited_hits and cand.j in cited_hits:
            merged.append(candidate_to_conflict(cand, cid_of_hit, by_id))
    return merged

