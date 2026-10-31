"""Local sentence embeddings: bge-small-en-v1.5 via fastembed (ONNX, CPU)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import config


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=str(config.ROOT / ".cache" / "fastembed"))


def encode(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Return L2-normalised float32 vectors, shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    if is_query:
        texts = [config.QUERY_PREFIX + t for t in texts]
    vectors = np.array(list(_model().embed(texts, batch_size=32)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)
