from pathlib import Path

from rag_assistant.ingestion import ingest_path

SAMPLE = Path(__file__).parent.parent / "data" / "sample"


def test_ingest_and_search(store):
    report = ingest_path(SAMPLE, store, chunk_size=400, overlap=60)
    assert report.files == 3
    assert report.chunks == store.count() > 3

    hits = store.search("cosine similarity between vectors", top_k=3)
    assert hits[0].source.endswith("vector-databases.md")
    assert hits[0].metadata["title"] == "Vector Databases"


def test_reingest_does_not_duplicate(store):
    ingest_path(SAMPLE, store)
    first = store.count()
    ingest_path(SAMPLE, store)
    assert store.count() == first


def test_reingest_removes_stale_chunks(store, tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("\n\n".join(f"Paragraph {i} " + "text " * 50 for i in range(10)))
    ingest_path(doc, store, chunk_size=300, overlap=0)
    many = store.count()

    doc.write_text("Now it is short.")
    ingest_path(doc, store, chunk_size=300, overlap=0)
    assert store.count() == 1 < many


def test_search_on_empty_store(store):
    assert store.search("anything") == []
