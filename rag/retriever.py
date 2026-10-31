"""Hybrid retrieval (FR-03): dense cosine + BM25, fused with Reciprocal Rank Fusion.

mode="dense" / "bm25" rank by one signal only (used by the eval ablation).
Every hit keeps its raw dense cosine and both ranks for the gate and debug panel.
"""

from __future__ import annotations

import numpy as np

import config
from common.schemas import Filters, RetrievalMode, ScoredChunk
from rag.embeddings import encode
from rag.index import Index


def _allowed(index: Index, filters: Filters | None) -> np.ndarray:
    mask = np.ones(len(index), dtype=bool)
    if not filters:
        return mask
    for i, c in enumerate(index.chunks):
        if filters.doc_ids and c.doc_id not in filters.doc_ids:
            mask[i] = False
        elif filters.version and c.version != filters.version:
            mask[i] = False
        elif filters.product and c.product != filters.product:
            mask[i] = False
    return mask


def _ranks(scores: np.ndarray, mask: np.ndarray, limit: int, positive_only: bool = False) -> dict[int, int]:
    order = [i for i in np.argsort(-scores) if mask[i] and (not positive_only or scores[i] > 0)]
    return {int(i): rank for rank, i in enumerate(order[:limit], start=1)}


def retrieve(index: Index, question: str, k: int = config.TOP_K, filters: Filters | None = None,
             mode: RetrievalMode = "hybrid") -> list[ScoredChunk]:
    if len(index) == 0:
        return []
    mask = _allowed(index, filters)
    dense = index.matrix @ encode([question], is_query=True)[0]
    bm25 = index.bm25_scores(question)
    dense_ranks = _ranks(dense, mask, config.DENSE_K)
    bm25_ranks = _ranks(bm25, mask, config.BM25_K, positive_only=True)

    rrf: dict[int, float] = {}
    for ranks in (dense_ranks, bm25_ranks):
        for i, rank in ranks.items():
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (config.RRF_K + rank)

    if mode == "dense":
        final = {i: float(dense[i]) for i in dense_ranks}
    elif mode == "bm25":
        final = {i: float(bm25[i]) for i in bm25_ranks}
    else:
        final = rrf

    top = sorted(final, key=lambda i: (-final[i], -dense[i]))[:k]
    return [
        ScoredChunk(
            chunk=index.chunks[i],
            dense_score=round(float(dense[i]), 4),
            dense_rank=dense_ranks.get(i),
            bm25_rank=bm25_ranks.get(i),
            rrf_score=round(rrf.get(i, 0.0), 5),
            final_score=round(final[i], 5),
        )
        for i in top
    ]
