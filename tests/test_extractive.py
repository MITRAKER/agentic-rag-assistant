from rag_assistant.answering import Source
from rag_assistant.answering.extractive import NOT_FOUND, ExtractiveAnswerer

SOURCES = [
    Source(
        "# Vector Databases\n\nQdrant uses HNSW, a layered graph. Payloads store metadata.",
        "vector.md",
        "Vector Databases",
        0.8,
    ),
    Source("Agents call tools. A router picks a source.", "agents.md", "AI Agents", 0.6),
]


def test_picks_matching_sentence_and_cites_it():
    answer = ExtractiveAnswerer().answer("What does Qdrant use?", SOURCES)
    assert answer.text.startswith("Qdrant uses HNSW, a layered graph.")
    assert answer.segments[0].citations[0].source_index == 0
    assert answer.model == "extractive"


def test_skips_headings():
    answer = ExtractiveAnswerer().answer("vector databases", SOURCES)
    assert "# Vector Databases" not in answer.text


def test_respects_max_sentences_and_reading_order():
    answer = ExtractiveAnswerer(max_sentences=2).answer("router tools Qdrant", SOURCES)
    assert len(answer.segments) == 2
    # reading order: the better-ranked source first
    assert [s.citations[0].source_index for s in answer.segments] == [0, 1]


def test_no_overlap_means_not_found():
    answer = ExtractiveAnswerer().answer("capital of France?", SOURCES)
    assert answer.text == NOT_FOUND
    assert answer.cited_source_indices() == []


def test_duplicate_sentences_from_overlapping_chunks_appear_once():
    dup = [
        Source("Qdrant uses HNSW. It is fast.", "v.md", "V", 0.9),
        Source("It is fast. Qdrant uses HNSW.", "v.md", "V", 0.8),
    ]
    answer = ExtractiveAnswerer().answer("Does Qdrant use HNSW?", dup)
    assert answer.text == "Qdrant uses HNSW."
    assert answer.segments[0].citations[0].source_index == 0
