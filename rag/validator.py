"""Citations (FR-05): validate the LLM draft, build Citation objects, pick supporting excerpts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.schemas import Citation, Claim, Conflict, ConflictPosition, ScoredChunk
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


# ---------------------------------------------------------------- LLM draft validation


@dataclass
class Validated:
    claims: list[Claim]
    citations: list[Citation]
    conflicts: list[Conflict]
    answer: str
    cid_of_hit: dict[int, int]  # hit index -> citation id
    dropped_claims: int = 0
    invalid_ids: list[str] = field(default_factory=list)


def _hit_index(ref, n_hits: int) -> int | None:
    match = re.fullmatch(r"\[?S?(\d+)\]?", str(ref).strip(), re.I)
    if not match:
        return None
    i = int(match.group(1)) - 1
    return i if 0 <= i < n_hits else None


def _renumber_refs(text: str, cid_of_hit: dict[int, int]) -> str:
    """Rewrite the model's internal evidence ids ("S2") into the user-facing citation numbers ("[1]")."""
    def sub(match: re.Match) -> str:
        cid = cid_of_hit.get(int(match.group(1)) - 1)
        return f"[{cid}]" if cid else "another source"
    return re.sub(r"\[?\bS(\d+)\b\]?", sub, text)


def _source_text(hit: ScoredChunk) -> str:
    c = hit.chunk
    return f"{c.embed_text} {c.filename} {c.version or ''} {c.effective_date or ''}"


def _support(claim: str, chunk_texts: list[str]) -> str:
    """'weak' when the claim's numbers are absent from its sources or word overlap is low."""
    source = " ".join(chunk_texts).lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", claim)
    if any(n not in source for n in numbers):
        return "weak"
    words = set(tokenize(claim))
    if words and len(words & set(tokenize(source))) / len(words) < 0.35:
        return "weak"
    return "supported"


def validate(draft: dict, hits: list[ScoredChunk]) -> Validated:
    """Drop invented evidence ids and uncited claims, renumber citations [1..n] by first use."""
    cid_of_hit: dict[int, int] = {}
    focus: dict[int, list[str]] = {}
    invalid: list[str] = []

    def cite(ref, text: str) -> int | None:
        i = _hit_index(ref, len(hits))
        if i is None:
            invalid.append(str(ref))
            return None
        cid_of_hit.setdefault(i, len(cid_of_hit) + 1)
        focus.setdefault(i, []).append(text)
        return cid_of_hit[i]

    raw_claims = [c for c in draft.get("claims") or [] if isinstance(c, dict) and str(c.get("text", "")).strip()]
    kept: list[tuple[str, list[int], list[int]]] = []
    for c in raw_claims:
        text = str(c["text"]).strip()
        refs = c.get("evidence") or c.get("citations") or []
        refs = refs if isinstance(refs, list) else [refs]
        ids, hit_ids = [], []
        for ref in refs:
            cid = cite(ref, text)
            if cid is not None and cid not in ids:
                ids.append(cid)
                hit_ids.append(_hit_index(ref, len(hits)))
        if ids:
            kept.append((text, ids, hit_ids))

    raw_conflicts = []
    for conf in draft.get("conflicts") or []:
        if not isinstance(conf, dict):
            continue
        positions = []
        for pos in conf.get("positions") or []:
            if isinstance(pos, dict) and (cid := cite(pos.get("evidence"), str(pos.get("statement", "")))) is not None:
                positions.append((cid, str(pos.get("statement", "")).strip()))
        if len({cid for cid, _ in positions}) >= 2:
            raw_conflicts.append((conf, positions))

    hit_of_cid = {cid: i for i, cid in cid_of_hit.items()}
    citations = [make_citation(cid, hits[hit_of_cid[cid]], " ".join(focus[hit_of_cid[cid]]))
                 for cid in sorted(hit_of_cid)]
    by_id = {c.id: c for c in citations}

    claims = [Claim(text=_renumber_refs(text, cid_of_hit), citation_ids=ids, support=_support(text, [_source_text(hits[i]) for i in hit_ids]))
              for text, ids, hit_ids in kept]
    conflicts = []
    for conf, positions in raw_conflicts:
        ctype = conf.get("type") if conf.get("type") in ("version_sensitive", "value_mismatch", "contradiction") else "contradiction"
        conflicts.append(Conflict(
            type=ctype,
            topic=_renumber_refs(str(conf.get("topic") or "Sources disagree"), cid_of_hit),
            summary=_renumber_refs(str(conf.get("summary") or ""), cid_of_hit),
            positions=[ConflictPosition(citation_id=cid, statement=_renumber_refs(st, cid_of_hit) or by_id[cid].excerpt[:80],
                                        version=by_id[cid].version, effective_date=by_id[cid].effective_date)
                       for cid, st in positions],
            resolution_hint=_renumber_refs(str(conf["resolution_hint"]), cid_of_hit) if conf.get("resolution_hint") else None,
            detected_by="llm",
        ))
    if not claims and conflicts:  # model put both sides only in conflicts[]: each position is a cited statement
        for conflict in conflicts:
            for pos in conflict.positions:
                c = by_id[pos.citation_id]
                label = c.doc_title + (f" v{c.version}" if c.version else "") + (f" ({c.effective_date})" if c.effective_date else "")
                claims.append(Claim(text=f"{label}: {pos.statement}", citation_ids=[pos.citation_id],
                                    support=_support(pos.statement, [_source_text(hits[hit_of_cid[pos.citation_id]])])))
    answer = " ".join(f"{re.sub(r'\s*\[\d+(?:,\s*\d+)*\]\s*$', '', cl.text).rstrip('.')} [{', '.join(map(str, cl.citation_ids))}]." for cl in claims)
    return Validated(claims, citations, conflicts, answer, cid_of_hit,
                     dropped_claims=len(raw_claims) - len(kept), invalid_ids=invalid)
