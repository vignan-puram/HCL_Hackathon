"""File -> (DocumentRecord, chunks): validate, parse, extract metadata, chunk.

Run from repo root to inspect one file: python -m ingestion.pipeline <path>
"""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from common.schemas import Block, Chunk, DocumentRecord, UserMeta
from ingestion.chunker import chunk_blocks
from ingestion.errors import IngestionError
from ingestion.parsers import DOC_TYPES, ParsedFile, parse_file

_DATE = r"(\d{4}-\d{2}-\d{2})"
_KIND_KEYWORDS = [
    ("faq", "faq"),
    ("release", "release_notes"),
    ("notes", "release_notes"),
    ("polic", "policy"),
    ("limits", "policy"),
    ("troubleshoot", "support_article"),
    ("support", "support_article"),
    ("guide", "manual"),
    ("manual", "manual"),
    ("reference", "manual"),
]
_VALID_KINDS = {"manual", "faq", "release_notes", "policy", "support_article", "other"}


def doc_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _extract_version(meta: dict[str, str], head: str) -> str | None:
    if meta.get("version"):
        return meta["version"]
    match = re.search(r"\bversion\s*:?\s*v?(\d+(?:\.\d+)*)", head, re.I) or re.search(r"\bv(\d+\.\d+(?:\.\d+)*)\b", head)
    return match.group(1) if match else None


def _extract_date(meta: dict[str, str], head: str) -> str | None:
    for key in ("effective_date", "effective-date", "date", "updated", "last-updated", "released"):
        found = re.search(_DATE, meta.get(key, ""))
        if found:
            return found.group(1)
    match = re.search(r"(?:effective|updated|revised|released|reviewed|date)[^0-9\n]{0,25}" + _DATE, head, re.I)
    if match:
        return match.group(1)
    match = re.search(_DATE, head[:400])
    return match.group(1) if match else None


def _extract_kind(meta: dict[str, str], filename: str, title: str) -> str:
    if meta.get("kind") in _VALID_KINDS:
        return meta["kind"]
    haystack = f"{filename} {title}".lower()
    for keyword, kind in _KIND_KEYWORDS:
        if keyword in haystack:
            return kind
    return "other"


def _strip_title(blocks: list[Block], title: str) -> list[Block]:
    """Drop the document title from the front of section paths (it is already the breadcrumb root)."""
    out = []
    for b in blocks:
        is_title = b.section_path and b.section_path[0].lower() == title.lower()
        path = b.section_path[1:] if is_title else b.section_path
        out.append(b.model_copy(update={"section_path": path}))
    return out


def build_record(path: Path, data: bytes, parsed: ParsedFile, user_meta: UserMeta | None,
                 filename: str, num_chunks: int = 0) -> DocumentRecord:
    meta = parsed.meta
    head = "\n".join(b.text for b in parsed.blocks[:8])[:2000]
    title = meta.get("title") or Path(filename).stem.replace("_", " ").title()
    if title.isupper():
        title = title.title()
    um = user_meta or UserMeta()
    return DocumentRecord(
        doc_id=doc_id_for(data),
        filename=filename,
        title=title,
        doc_type=DOC_TYPES[path.suffix.lower()],
        doc_kind=um.doc_kind or _extract_kind(meta, filename, title),
        version=um.version or _extract_version(meta, head),
        effective_date=um.effective_date or _extract_date(meta, head),
        product=um.product,
        pages=parsed.pages,
        num_chunks=num_chunks,
        uploaded_at=datetime.now(timezone.utc),
        source_url=filename,
    )


def process_file(path: Path, user_meta: UserMeta | None = None,
                 filename: str | None = None) -> tuple[DocumentRecord, list[Chunk]]:
    """Parse, clean, extract metadata and chunk one file.

    filename overrides the display name (uploads are saved under a temp name).
    Raises IngestionError: UNSUPPORTED_FORMAT, FILE_TOO_LARGE, EMPTY_DOCUMENT, PARSE_FAILED.
    """
    path = Path(path)
    filename = filename or path.name
    suffix = path.suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise IngestionError("UNSUPPORTED_FORMAT", f"'{suffix or filename}' is not supported. Use PDF, HTML, Markdown or TXT.")
    data = path.read_bytes()
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise IngestionError("FILE_TOO_LARGE", f"File is larger than {config.MAX_FILE_MB} MB.")
    if not data.strip():
        raise IngestionError("EMPTY_DOCUMENT", "The file is empty.")

    try:
        parsed = parse_file(path, DOC_TYPES[suffix])
    except Exception as exc:  # corrupt file, bad markup, encrypted PDF ...
        raise IngestionError("PARSE_FAILED", f"Could not parse the file: {type(exc).__name__}: {exc}") from exc
    if not any(b.text.strip() for b in parsed.blocks):
        hint = " Scanned (image-only) PDFs are not supported." if suffix == ".pdf" else ""
        raise IngestionError("EMPTY_DOCUMENT", "No extractable text." + hint)

    record = build_record(path, data, parsed, user_meta, filename)
    blocks = _strip_title(parsed.blocks, record.title)
    chunks = chunk_blocks(blocks, record)
    return record.model_copy(update={"num_chunks": len(chunks)}), chunks


def _main(paths: list[str]) -> None:
    for p in paths:
        record, chunks = process_file(Path(p))
        print(f"\n== {record.filename} | {record.title} | {record.doc_type}/{record.doc_kind} "
              f"| v={record.version} | date={record.effective_date} | pages={record.pages} | chunks={len(chunks)}")
        for c in chunks:
            page = f" p{c.page_start}" if c.page_start else ""
            print(f"  [{c.ordinal:02d}] {' > '.join(c.section_path) or '(preamble)'}{page} "
                  f"({c.word_count}w): {c.text[:70]!r}")


if __name__ == "__main__":
    _main(sys.argv[1:] or [str(p) for p in sorted(config.CORPUS_DIR.iterdir()) if p.suffix in config.ALLOWED_EXTENSIONS])
