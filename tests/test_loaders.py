from pathlib import Path

import pytest

from rag_assistant.ingestion.loaders import load_file, load_path

SAMPLE = Path(__file__).parent.parent / "data" / "sample"


def test_markdown_title_from_heading():
    [doc] = load_file(SAMPLE / "rag-basics.md")
    assert doc.metadata["title"] == "Retrieval-Augmented Generation (RAG)"
    assert "vector database" in doc.text


def test_html_strips_scripts(tmp_path):
    page = tmp_path / "page.html"
    page.write_text(
        "<html><head><title>T</title><script>evil()</script></head>"
        "<body><p>Real content</p></body></html>"
    )
    [doc] = load_file(page)
    assert doc.metadata["title"] == "T"
    assert "Real content" in doc.text
    assert "evil" not in doc.text


def test_load_directory_finds_supported_files():
    docs = load_path(SAMPLE)
    assert {Path(d.source).name for d in docs} == {
        "rag-basics.md",
        "vector-databases.md",
        "ai-agents.md",
    }


def test_unsupported_type(tmp_path):
    f = tmp_path / "x.docx"
    f.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(f)


def test_too_large_file_is_rejected(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 100)
    with pytest.raises(ValueError, match="limit"):
        load_file(f, max_bytes=10)


def test_too_many_files_is_rejected(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.md").write_text("hi")
    with pytest.raises(ValueError, match="More than 2"):
        load_path(tmp_path, max_files=2)


def test_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "nope")


def test_pdf_has_one_document_per_page(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf = tmp_path / "blank.pdf"
    with pdf.open("wb") as fh:
        writer.write(fh)
    assert load_file(pdf) == []  # blank pages have no text, so they are skipped
