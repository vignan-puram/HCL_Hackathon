"""Section-aware chunking (FR-02).

Rules, in order:
1. A chunk never crosses a top-level section (section_path[0]).
2. Each (sub)section becomes its own chunk when it fits (<= CHUNK_MAX_WORDS);
   longer sections split on paragraphs, then sentences, with word overlap.
3. Sections under MIN_SECTION_WORDS merge into the next sibling under the same
   top-level section (or the previous one when there is no next sibling).
4. Tables and code blocks are never split.
5. embed_text = "{doc_title} > {section path}\\n{text}" so headings are searchable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from common.schemas import Block, Chunk, DocumentRecord


@dataclass
class _Section:
    path: list[str]
    blocks: list[Block] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)

    @property
    def top(self) -> str:
        return self.path[0] if self.path else ""


def _sections(blocks: list[Block]) -> list[_Section]:
    sections: list[_Section] = []
    for block in blocks:
        if not sections or sections[-1].path != block.section_path:
            sections.append(_Section(path=list(block.section_path)))
        sections[-1].blocks.append(block)
    return sections


def _labelled(sec: _Section) -> list[Block]:
    """Blocks of a merged section, with its leaf heading kept as the first line."""
    if not sec.path or not sec.blocks:
        return sec.blocks
    first = sec.blocks[0]
    return [first.model_copy(update={"text": f"{sec.path[-1]}\n{first.text}"})] + sec.blocks[1:]


def _merge_small(sections: list[_Section]) -> list[_Section]:
    merged: list[_Section] = []
    carry: _Section | None = None
    for i, sec in enumerate(sections):
        if carry is not None:  # carry shares sec's top-level section by construction
            common = _common_path(carry.path, sec.path) or sec.path
            own = _labelled(sec) if len(common) < len(sec.path) else sec.blocks
            sec = _Section(path=common, blocks=_labelled(carry) + own)
            carry = None
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        if sec.words < config.MIN_SECTION_WORDS:
            if nxt is not None and nxt.top == sec.top:
                carry = sec
                continue
            prev = merged[-1] if merged else None
            if prev is not None and prev.top == sec.top and prev.words + sec.words <= config.CHUNK_MAX_WORDS:
                common = _common_path(prev.path, sec.path) or prev.path
                if len(common) < len(prev.path):
                    prev.blocks = _labelled(prev)
                prev.path = common
                prev.blocks.extend(_labelled(sec))
                continue
        merged.append(sec)
    return merged


def _common_path(a: list[str], b: list[str]) -> list[str]:
    common = []
    for x, y in zip(a, b):
        if x != y:
            break
        common.append(x)
    return common


def _split_long(text: str, max_words: int) -> list[str]:
    """Split one paragraph that alone exceeds max_words on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack(blocks: list[Block]) -> list[list[Block]]:
    """Group blocks into pieces of ~CHUNK_TARGET_WORDS with CHUNK_OVERLAP_WORDS overlap."""
    units: list[Block] = []
    for b in blocks:
        is_atomic = "\n" in b.text and (" | " in b.text or b.text.startswith("```"))
        if len(b.text.split()) > config.CHUNK_MAX_WORDS and not is_atomic:
            units.extend(b.model_copy(update={"text": t}) for t in _split_long(b.text, config.CHUNK_TARGET_WORDS))
        else:
            units.append(b)

    total = sum(len(u.text.split()) for u in units)
    if total <= config.CHUNK_MAX_WORDS:
        return [units]

    pieces, current, words = [], [], 0
    for unit in units:
        n = len(unit.text.split())
        if current and words + n > config.CHUNK_TARGET_WORDS:
            pieces.append(current)
            tail = " ".join(" ".join(u.text for u in current).split()[-config.CHUNK_OVERLAP_WORDS:])
            current = [current[-1].model_copy(update={"text": "... " + tail})]
            words = len(tail.split())
        current.append(unit)
        words += n
    if current:
        pieces.append(current)
    return pieces


def chunk_blocks(blocks: list[Block], doc: DocumentRecord) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _merge_small(_sections(blocks)):
        for piece in _pack(section.blocks):
            text = "\n".join(b.text for b in piece).strip()
            if not text:
                continue
            pages = [b.page for b in piece if b.page is not None]
            breadcrumb = " > ".join([doc.title] + section.path)
            ordinal = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}:{ordinal:04d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                filename=doc.filename,
                doc_type=doc.doc_type,
                doc_kind=doc.doc_kind,
                product=doc.product,
                section_path=section.path,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                ordinal=ordinal,
                text=text,
                embed_text=f"{breadcrumb}\n{text}",
                version=doc.version,
                effective_date=doc.effective_date,
                word_count=len(text.split()),
            ))
    return chunks
