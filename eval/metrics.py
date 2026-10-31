"""Evaluation metrics (FR-08).

A retrieved chunk is relevant to a gold item when it comes from the gold document
and contains the gold evidence string (case- and whitespace-insensitive). Labels
therefore survive any change to chunk size or chunk ids.
"""

from __future__ import annotations

import re
from collections import defaultdict

from common.schemas import Chunk, QueryResult

ANSWERABLE = {"direct", "paraphrase", "terminology", "conflict", "scoped"}
K_VALUES = (1, 3, 5)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, gold: dict) -> bool:
    return chunk.filename == gold["doc"] and _norm(gold["evidence"]) in _norm(chunk.text)


def retrieval_scores(chunks: list[Chunk], gold: list[dict]) -> dict[str, float]:
    """Recall@k, Hit@k, Precision@k and reciprocal rank for one question."""
    out: dict[str, float] = {}
    first = next((r for r, c in enumerate(chunks, start=1) if any(is_relevant(c, g) for g in gold)), None)
    for k in K_VALUES:
        top = chunks[:k]
        found = sum(1 for g in gold if any(is_relevant(c, g) for c in top))
        out[f"recall@{k}"] = found / len(gold)
        out[f"hit@{k}"] = float(found > 0)
        out[f"precision@{k}"] = sum(1 for c in top if any(is_relevant(c, g) for g in gold)) / k
    out["mrr"] = 1.0 / first if first else 0.0
    return out


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: round(sum(r[key] for r in rows) / len(rows), 3) for key in rows[0]}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


def behaviour_metrics(items: list[dict], results: list[QueryResult], chunk_text: dict[str, str]) -> dict:
    """Status, fallback, conflict and citation metrics over all questions."""
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_cat: dict[str, list[tuple[dict, QueryResult]]] = defaultdict(list)
    for item, res in zip(items, results):
        confusion[item["expected_status"]][res.status] += 1
        by_cat[item["category"]].append((item, res))

    def rate(pairs, predicate) -> float:
        return round(sum(1 for p in pairs if predicate(*p)) / len(pairs), 3) if pairs else 0.0

    answerable = [p for c in ANSWERABLE for p in by_cat[c]]
    non_conflict = [p for c in ANSWERABLE - {"conflict"} for p in by_cat[c]]
    both_sides = lambda item, res: {g["doc"] for g in item["gold"]} <= {c.filename for c in res.citations}  # noqa: E731

    cited, cited_ok, coverage = 0, 0, []
    for item, res in answerable:
        if res.status in ("not_found", "needs_clarification"):
            continue
        for c in res.citations:
            cited += 1
            text = _norm(chunk_text.get(c.chunk_id, ""))
            cited_ok += any(c.filename == g["doc"] and _norm(g["evidence"]) in text for g in item["gold"])
        if item["answer_keywords"]:
            answer = res.answer.lower()
            coverage.append(sum(k.lower() in answer for k in item["answer_keywords"]) / len(item["answer_keywords"]))

    totals = [r.timings_ms.get("total", 0) for r in results]
    return {
        "questions": len(results),
        "status_accuracy": rate(list(zip(items, results)), lambda i, r: r.status == i["expected_status"]),
        "not_found_recall": rate(by_cat["unanswerable"], lambda i, r: r.status == "not_found"),
        "false_refusal_rate": rate(answerable, lambda i, r: r.status in ("not_found", "needs_clarification")),
        "clarification_rate": rate(by_cat["ambiguous"], lambda i, r: r.status == "needs_clarification"),
        "conflict_recall": rate(by_cat["conflict"], lambda i, r: r.status == "conflict" and both_sides(i, r)),
        "conflict_false_positive_rate": rate(non_conflict, lambda i, r: r.status == "conflict"),
        "citation_precision": round(cited_ok / cited, 3) if cited else 0.0,
        "answer_keyword_coverage": round(sum(coverage) / len(coverage), 3) if coverage else 0.0,
        "claims_removed_by_validator": sum((r.debug or {}).get("validator", {}).get("dropped_claims", 0) for r in results),
        "llm_calls_avoided_by_gate": sum(1 for r in results if not (r.debug or {}).get("llm_called")),
        "degraded_answers": sum(1 for r in results if r.degraded),
        "latency_ms_p50": percentile(totals, 50),
        "latency_ms_p95": percentile(totals, 95),
        "confusion": {exp: dict(pred) for exp, pred in confusion.items()},
    }
