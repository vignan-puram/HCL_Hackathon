"""Grounded answer generation (FR-04): numbered-evidence prompt, one JSON-mode LLM call."""

from __future__ import annotations

import json
import re

from common.schemas import ScoredChunk
from rag import llm
from rag.conflicts import ConflictCandidate

SYSTEM_PROMPT = """You are an enterprise documentation assistant. You answer ONLY from the numbered evidence passages you are given.

Rules:
1. Use only facts stated in the evidence. Never use outside knowledge and never guess.
2. Every claim is one short sentence and cites at least one evidence id, e.g. ["S1"]. Use at most 5 claims.
3. If passages disagree about the same thing (different numbers, versions or dates), state each position as a separate claim that names its source document and version/date, and add an entry to "conflicts". Never silently pick one side. When dates differ, mention which source is newer in resolution_hint.
4. Passages that apply to different plans, products or scopes are NOT conflicts: explain the scope instead.
5. If the evidence does not answer the question, return status "not_found" with no claims.
6. If the question is too vague to answer because it could mean several different things, return status "needs_clarification" and one short clarifying_question that lists the options found in the evidence.
7. If the evidence answers only part of the question, return status "partial" and describe what is missing in missing_info.

Return only JSON with this shape:
{"status": "answered" | "partial" | "conflict" | "needs_clarification" | "not_found",
 "claims": [{"text": "...", "evidence": ["S1"]}],
 "conflicts": [{"type": "version_sensitive" | "value_mismatch" | "contradiction", "topic": "...", "summary": "...",
                "positions": [{"evidence": "S1", "statement": "..."}, {"evidence": "S2", "statement": "..."}],
                "resolution_hint": "..."}],
 "clarifying_question": null,
 "missing_info": null}"""


def format_evidence(hits: list[ScoredChunk]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        header = [c.doc_title, f"section: {' > '.join(c.section_path) or 'introduction'}"]
        if c.page_start:
            header.append(f"page {c.page_start}")
        header.append(f"version {c.version}" if c.version else "version n/a")
        header.append(f"date {c.effective_date}" if c.effective_date else "date n/a")
        blocks.append(f"[S{i}] {' | '.join(header)}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[ScoredChunk], hints: list[ConflictCandidate]) -> str:
    parts = [f"Question: {question}", "", "Evidence:", format_evidence(hits)]
    if hints:
        parts += ["", "Automatic checks flagged these possible conflicts. Confirm them if the passages really "
                      "disagree about the same thing, otherwise ignore them:"]
        parts += [f"- S{h.i + 1} vs S{h.j + 1}: {h.summary}" for h in hints]
    return "\n".join(parts)


def parse_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def generate(question: str, hits: list[ScoredChunk], hints: list[ConflictCandidate]) -> tuple[dict, llm.LLMResponse]:
    """Return (parsed draft, raw response). Raises llm.LLMError if no valid JSON can be obtained."""
    user = build_user_prompt(question, hits, hints)
    resp = llm.complete_json(SYSTEM_PROMPT, user)
    draft = parse_json(resp.text)
    if draft is None:  # one repair attempt
        resp = llm.complete_json(SYSTEM_PROMPT, user + "\n\nYour previous reply was not valid JSON. Return only valid JSON.",
                                 use_cache=False)
        draft = parse_json(resp.text)
        if draft is None:
            raise llm.LLMError("model returned invalid JSON twice")
    return draft, resp
