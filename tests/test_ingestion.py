"""Ingestion: parsers per format, metadata, chunk boundaries, errors."""

from pathlib import Path

import pytest

import config
from ingestion.errors import IngestionError
from ingestion.pipeline import process_file

CORPUS = config.CORPUS_DIR


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_markdown_headings_and_front_matter(tmp_path):
    path = _write(tmp_path, "guide.md", "---\ntitle: Guide\nversion: \"1.5\"\n---\n# Guide\n\n## Setup\n\nInstall it.\n\n### Linux\n\nUse apt.\n")
    record, chunks = process_file(path)
    assert (record.title, record.version, record.doc_type) == ("Guide", "1.5", "md")
    assert chunks[0].section_path[0] == "Setup"
    assert chunks[0].embed_text.startswith("Guide > Setup")


def test_html_drops_noise_and_keeps_tables(tmp_path):
    html = ("<html><head><title>T</title><script>evil()</script></head><body><nav>Menu</nav>"
            "<h1>T</h1><h2>Limits</h2><p>Body text.</p><table><tr><td>Free</td><td>2 GB</td></tr></table>"
            "<footer>Copyright</footer></body></html>")
    _, chunks = process_file(_write(tmp_path, "t.html", html))
    text = " ".join(c.text for c in chunks)
    assert "Free | 2 GB" in text and "evil" not in text and "Menu" not in text and "Copyright" not in text


def test_txt_heading_styles(tmp_path):
    txt = "My Doc\n======\n\n1. Scope\n\nThis is the scope.\n\nERRORS AND FIXES\n\nFix things.\n"
    record, chunks = process_file(_write(tmp_path, "doc.txt", txt))
    assert record.title == "My Doc"
    paths = {tuple(c.section_path) for c in chunks}
    assert ("1. Scope",) in paths or any("1. Scope" in c.text for c in chunks)


def test_pdf_pages_headings_and_header_stripping():
    record, chunks = process_file(CORPUS / "admin_guide_v3.pdf")
    assert (record.version, record.effective_date, record.pages) == ("3.0", "2025-02-15", 4)
    reset = next(c for c in chunks if "reset link expires" in c.text)
    assert reset.section_path == ["3. Resetting a User's Password"] and reset.page_start == 3
    assert not any("Nimbus Labs (fictional) - Page" in c.text for c in chunks)


@pytest.mark.parametrize("name,version,date", [
    ("billing_policy_v2.pdf", "2.0", "2025-03-01"),
    ("billing_faq.md", None, "2023-06-10"),
    ("api_reference_v2.html", "2.4", "2023-09-12"),
    ("release_notes_3_0.md", "3.0", "2025-04-20"),
])
def test_conflict_documents_metadata(name, version, date):
    record, _ = process_file(CORPUS / name)
    assert (record.version, record.effective_date) == (version, date)


def test_whole_corpus_chunks_respect_limits():
    for path in CORPUS.iterdir():
        if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue
        record, chunks = process_file(path)
        assert chunks, path.name
        assert all(c.word_count <= config.CHUNK_MAX_WORDS for c in chunks)
        assert all(c.chunk_id.startswith(record.doc_id) for c in chunks)


def test_long_section_is_split_with_overlap(tmp_path):
    sentences = " ".join(f"Sentence number {i} explains a detail of the policy." for i in range(120))
    _, chunks = process_file(_write(tmp_path, "long.md", f"# Long\n\n## Part\n\n{sentences}\n"))
    assert len(chunks) >= 2
    assert all(c.section_path == ["Part"] for c in chunks)
    assert all(c.word_count <= config.CHUNK_MAX_WORDS for c in chunks)


def test_errors(tmp_path):
    with pytest.raises(IngestionError) as e:
        process_file(_write(tmp_path, "x.docx", "hello"))
    assert e.value.code == "UNSUPPORTED_FORMAT"
    with pytest.raises(IngestionError) as e:
        process_file(_write(tmp_path, "empty.txt", "   \n"))
    assert e.value.code == "EMPTY_DOCUMENT"
