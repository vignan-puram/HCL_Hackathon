"""File -> (DocumentRecord, chunks). Owner: Member 2.

STUB (Gate A): signature is final; Member 2 replaces the body.
"""

from __future__ import annotations

from pathlib import Path

from common.schemas import Chunk, DocumentRecord, UserMeta


def process_file(path: Path, user_meta: UserMeta | None = None) -> tuple[DocumentRecord, list[Chunk]]:
    """Parse, clean, extract metadata and chunk one file.

    Raises ingestion.errors.IngestionError with code UNSUPPORTED_FORMAT,
    EMPTY_DOCUMENT or PARSE_FAILED.
    """
    raise NotImplementedError("process_file: Member 2, Build 1")
