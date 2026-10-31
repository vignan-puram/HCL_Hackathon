"""In-memory index: chunk store + dense NumPy matrix + BM25, persisted to disk.

Files in config.INDEX_DIR: documents.json, chunks.jsonl, embeddings.npy.
At hackathon scale (< 10k chunks) brute-force cosine takes milliseconds,
so no vector database is needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from common.schemas import Chunk, DocumentRecord

_STOPWORDS = set(
    "a an the and or of to in on for with by is are was were be been it its this that these those "
    "do does did how what when where which who why can could should would will i my me we our you your "
    "there their they them at as from if not no s".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; hyphenated codes like ERR-429 are kept whole and also split."""
    tokens = []
    for tok in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower()):
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok:
            tokens.extend(part for part in tok.split("-") if part and part not in _STOPWORDS)
    return tokens


class Index:
    def __init__(self) -> None:
        self.docs: dict[str, DocumentRecord] = {}
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 384), dtype=np.float32)
        self.bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, doc: DocumentRecord, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.delete(doc.doc_id)
        self.docs[doc.doc_id] = doc
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)]) if len(self.matrix) else vectors.astype(np.float32)
        self._rebuild_bm25()

    def delete(self, doc_id: str) -> None:
        if doc_id not in self.docs and not any(c.doc_id == doc_id for c in self.chunks):
            return
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.docs.pop(doc_id, None)
        self.chunks = [self.chunks[i] for i in keep]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, self.matrix.shape[1]), dtype=np.float32)
        self._rebuild_bm25()

    def clear(self) -> None:
        self.__init__()

    def _rebuild_bm25(self) -> None:
        corpus = [tokenize(c.embed_text) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def bm25_scores(self, query: str) -> np.ndarray:
        if self.bm25 is None:
            return np.zeros(0)
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        docs = [d.model_dump(mode="json") for d in self.docs.values()]
        (directory / "documents.json").write_text(json.dumps(docs, indent=1), encoding="utf-8")
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(c.model_dump_json() + "\n")
        np.save(directory / "embeddings.npy", self.matrix)

    def load(self, directory: Path) -> None:
        self.clear()
        if not (directory / "chunks.jsonl").exists():
            return
        docs = json.loads((directory / "documents.json").read_text(encoding="utf-8"))
        self.docs = {d["doc_id"]: DocumentRecord.model_validate(d) for d in docs}
        with open(directory / "chunks.jsonl", encoding="utf-8") as fh:
            self.chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        self.matrix = np.load(directory / "embeddings.npy")
        if len(self.matrix) != len(self.chunks):  # corrupted / partial write: start clean
            self.clear()
            return
        self._rebuild_bm25()
