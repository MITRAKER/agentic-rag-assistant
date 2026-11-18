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
    with pytest.raises(ValueError):
        load_file(f)
