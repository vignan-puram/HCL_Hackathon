"""Format parsers (FR-01): each turns a file into ordered Blocks plus raw metadata.

Every parser returns ParsedFile(blocks, meta, pages). section_path holds the
heading trail of each block; the document title heading is reported in
meta["title"] and stripped from paths later by the pipeline.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from common.schemas import Block


@dataclass
class ParsedFile:
    blocks: list[Block]
    meta: dict[str, str] = field(default_factory=dict)
    pages: int | None = None


def clean_text(text: str) -> str:
    """NFKC, drop soft hyphens, join words hyphenated across lines, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


class _SectionStack:
    """Tracks the heading trail; a new heading pops headings of equal or deeper level."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))

    @property
    def path(self) -> list[str]:
        return [title for _, title in self._stack]


# ---------------------------------------------------------------- Markdown


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, text[match.end():]


def parse_md(text: str) -> ParsedFile:
    meta, body = _front_matter(text)
    stack, blocks, para = _SectionStack(), [], []
    in_fence, fence = False, []

    def flush() -> None:
        if para:
            blocks.append(Block(text=clean_text(" ".join(para)), section_path=stack.path))
            para.clear()

    for line in body.splitlines():
        if line.strip().startswith("```"):
            if in_fence:  # keep code fences whole and verbatim
                fence.append(line)
                blocks.append(Block(text="\n".join(fence), section_path=stack.path))
                in_fence, fence = False, []
            else:
                flush()
                in_fence, fence = True, [line]
            continue
        if in_fence:
            fence.append(line)
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level, title = len(heading.group(1)), clean_text(heading.group(2))
            if level == 1 and "title" not in meta:
                meta["title"] = title
            stack.push(level, title)
        elif not line.strip():
            flush()
        else:
            para.append(re.sub(r"^\s*([-*+]|\d+\.)\s+", "- ", line) if re.match(r"^\s*([-*+]|\d+\.)\s", line) else line)
    flush()
    return ParsedFile(blocks, meta)


# ---------------------------------------------------------------- HTML

_HTML_NOISE = ["script", "style", "nav", "header", "footer", "noscript", "aside", "form"]
_HTML_CONTENT = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]


def parse_html(text: str) -> ParsedFile:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(text, "html.parser")
    meta: dict[str, str] = {}
    if soup.title and soup.title.string:
        meta["title"] = clean_text(soup.title.string)
    for tag in soup.find_all("meta"):
        if tag.get("name") and tag.get("content"):
            meta[tag["name"].lower()] = tag["content"].strip()
    for tag in soup(_HTML_NOISE):
        tag.decompose()

    stack, blocks = _SectionStack(), []
    root = soup.body or soup
    for el in root.find_all(_HTML_CONTENT):
        if el.find_parent(["table", "li", "pre", "p"]):
            continue  # already emitted as part of its container
        if el.name[0] == "h" and el.name[1:].isdigit():
            title = clean_text(el.get_text(" "))
            if el.name == "h1" and "title" not in meta:
                meta["title"] = title
            stack.push(int(el.name[1]), title)
        elif el.name == "table":
            rows = [
                " | ".join(clean_text(c.get_text(" ")) for c in row.find_all(["th", "td"]))
                for row in el.find_all("tr")
            ]
            if rows:
                blocks.append(Block(text="\n".join(rows), section_path=stack.path))
        else:
            content = el.get_text("\n" if el.name == "pre" else " ")
            content = content.strip() if el.name == "pre" else clean_text(content)
            if content:
                blocks.append(Block(text=("- " if el.name == "li" else "") + content, section_path=stack.path))
    return ParsedFile(blocks, meta)


# ---------------------------------------------------------------- TXT

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].{0,80})$")


def _txt_heading(line: str, next_line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    if re.fullmatch(r"=+", next_line.strip()) and len(next_line.strip()) >= 3:
        return 1, stripped
    if re.fullmatch(r"-+", next_line.strip()) and len(next_line.strip()) >= 3:
        return 2, stripped
    numbered = _NUMBERED.match(stripped)
    if numbered and not stripped.endswith("."):
        return 2 + numbered.group(1).count("."), stripped
    letters = re.sub(r"[^A-Za-z]", "", stripped)
    if len(letters) >= 4 and letters.isupper():
        return 2, stripped
    return None


def parse_txt(text: str) -> ParsedFile:
    lines = text.splitlines()
    stack, blocks, para, meta = _SectionStack(), [], [], {}

    def flush() -> None:
        if para:
            blocks.append(Block(text=clean_text(" ".join(para)), section_path=stack.path))
            para.clear()

    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:  # underline of the previous heading
            skip_next = False
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        heading = _txt_heading(line, nxt)
        if heading:
            flush()
            level, title = heading
            skip_next = bool(re.fullmatch(r"[=-]{3,}", nxt.strip()))
            if not blocks and not stack.path and "title" not in meta:
                meta["title"] = title  # first heading before any content is the document title
                level = 1
            stack.push(level, title)
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return ParsedFile(blocks, meta)


# ---------------------------------------------------------------- PDF


def _pdf_lines(doc) -> list[dict]:
    lines = []
    for page_no, page in enumerate(doc, start=1):
        height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = clean_text("".join(s["text"] for s in spans))
                lines.append({
                    "text": text,
                    "size": round(max(s["size"] for s in spans), 1),
                    "bold": any("bold" in s["font"].lower() or s["flags"] & 16 for s in spans),
                    "page": page_no,
                    "y0": line["bbox"][1],
                    "y1": line["bbox"][3],
                    "edge": line["bbox"][1] < 50 or line["bbox"][3] > height - 50,
                })
    return lines


def _strip_repeated(lines: list[dict], pages: int) -> list[dict]:
    """Drop header/footer lines that repeat (ignoring digits) on more than half the pages."""
    if pages < 2:
        return lines
    key = lambda ln: re.sub(r"\d+", "#", ln["text"])  # noqa: E731
    counts = Counter(key(ln) for ln in lines if ln["edge"])
    repeated = {k for k, n in counts.items() if n > pages / 2}
    return [ln for ln in lines if not (ln["edge"] and key(ln) in repeated)]


def parse_pdf(path: Path) -> ParsedFile:
    import pymupdf

    with pymupdf.open(path) as doc:
        pages = doc.page_count
        lines = _strip_repeated(_pdf_lines(doc), pages)
        pdf_title = (doc.metadata or {}).get("title", "")
    if not lines:
        return ParsedFile([], {}, pages)

    body_size = statistics.median(ln["size"] for ln in lines for _ in range(max(1, len(ln["text"]) // 20)))
    heading_sizes = sorted({ln["size"] for ln in lines if ln["size"] >= body_size * 1.15}, reverse=True)
    level_of = {size: i + 1 for i, size in enumerate(heading_sizes)}

    meta: dict[str, str] = {"title": pdf_title} if pdf_title else {}
    stack, blocks, para = _SectionStack(), [], []
    prev = None

    def flush() -> None:
        if para:
            blocks.append(Block(text=clean_text(" ".join(t for t, _ in para)), section_path=stack.path, page=para[0][1]))
            para.clear()

    for ln in lines:
        is_heading = ln["size"] in level_of and len(ln["text"]) < 120
        if is_heading:
            flush()
            level = level_of[ln["size"]]
            if level == 1 and not blocks and not stack.path:
                meta.setdefault("title", ln["text"])
                if ln["text"] == meta["title"]:
                    prev = ln
                    continue  # title is metadata, not a section
            stack.push(level, ln["text"])
        else:
            new_para = (
                prev is None
                or prev["page"] != ln["page"]
                or ln["y0"] - prev["y1"] > ln["size"] * 0.9
            )
            if new_para:
                flush()
            para.append((ln["text"], ln["page"]))
        prev = ln
    flush()
    return ParsedFile(blocks, meta, pages)


# ---------------------------------------------------------------- dispatch

DOC_TYPES = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "md", ".markdown": "md", ".txt": "txt"}


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_file(path: Path, doc_type: str) -> ParsedFile:
    if doc_type == "pdf":
        return parse_pdf(path)
    text = read_text(path)
    return {"html": parse_html, "md": parse_md, "txt": parse_txt}[doc_type](text)
