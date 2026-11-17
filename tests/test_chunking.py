from itertools import pairwise

import pytest

from rag_assistant.ingestion.chunking import chunk_document, split_text
from rag_assistant.ingestion.loaders import Document


def test_short_text_is_one_chunk():
    assert split_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = split_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_no_text_is_lost():
    words = [f"w{i}" for i in range(500)]
    chunks = split_text(" ".join(words), chunk_size=200, overlap=40)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)


def test_consecutive_chunks_overlap():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = split_text(text, chunk_size=200, overlap=50)
    for a, b in pairwise(chunks):
        # the start of each chunk should repeat something from the end of the previous one
        assert b.split()[0] in a


def test_prefers_paragraph_boundaries():
    text = "A" * 100 + "\n\n" + "B" * 100
    assert split_text(text, chunk_size=150, overlap=0) == ["A" * 100, "B" * 100]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("x", chunk_size=10, overlap=10)


def test_chunk_ids_are_stable_and_unique():
    doc = Document(text="one two three " * 100, source="a.md", metadata={"page": 1})
    first = chunk_document(doc, chunk_size=100, overlap=10)
    second = chunk_document(doc, chunk_size=100, overlap=10)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert all(c.metadata == {"page": 1} for c in first)
