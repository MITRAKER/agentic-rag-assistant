from fakes import FakeClient, cited_text, message
from rag_assistant.answering import Source
from rag_assistant.answering.claude import FALLBACK_BETA, ClaudeAnswerer, search_result_block

SOURCES = [
    Source("Qdrant uses HNSW. It is fast.", "docs/vector.md", "Vector Databases", 0.81),
    Source("RAG retrieves passages first.", "docs/report.pdf", "Report", 0.62, {"page": 3}),
]


def test_request_shape():
    client = FakeClient(message([{"type": "text", "text": "ok"}]))
    ClaudeAnswerer(client, effort="low").answer("What does Qdrant use?", SOURCES)
    [req] = client.requests

    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default"
    assert req["betas"] == [FALLBACK_BETA]
    assert "search results" in req["system"]

    [msg] = req["messages"]
    assert msg["role"] == "user"
    *results, question = msg["content"]
    assert question == {"type": "text", "text": "What does Qdrant use?"}
    assert [r["type"] for r in results] == ["search_result", "search_result"]
    # citations must be enabled on every search result, or the API rejects the request
    assert all(r["citations"] == {"enabled": True} for r in results)


def test_fallbacks_can_be_turned_off():
    client = FakeClient(message([{"type": "text", "text": "ok"}]))
    ClaudeAnswerer(client, fallbacks=False).answer("q", SOURCES)
    assert "fallbacks" not in client.requests[0]
    assert "betas" not in client.requests[0]


def test_search_result_block_has_one_text_block_per_sentence():
    block = search_result_block(SOURCES[0])
    assert block["source"] == "docs/vector.md"
    assert block["title"] == "Vector Databases"
    assert block["content"] == [
        {"type": "text", "text": "Qdrant uses HNSW."},
        {"type": "text", "text": "It is fast."},
    ]


def test_search_result_block_mentions_pdf_page():
    block = search_result_block(SOURCES[1])
    assert block["source"] == "docs/report.pdf#page=3"
    assert block["title"] == "Report (page 3)"


def test_parses_citations_into_segments():
    response = message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            cited_text("Qdrant uses HNSW.", 0, "Qdrant uses HNSW."),
            {"type": "text", "text": " Also, "},
            cited_text("RAG retrieves passages first.", 1, "RAG retrieves passages first."),
        ]
    )
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES)

    assert answer.render() == ("Qdrant uses HNSW. [1] Also, RAG retrieves passages first. [2]")
    assert answer.cited_source_indices() == [0, 1]
    assert answer.segments[0].citations[0].cited_text == "Qdrant uses HNSW."
    assert answer.model == "claude-opus-5-5"
    assert (answer.input_tokens, answer.output_tokens) == (1200, 80)
    assert not answer.refused and not answer.truncated


def test_ignores_citations_to_unknown_results():
    response = message([cited_text("Odd.", 7, "Odd.")])
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES)
    assert answer.segments[0].citations == []


def test_refusal_discards_partial_text():
    response = message(
        [{"type": "text", "text": "partial"}],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": None, "explanation": "Not allowed."},
    )
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES)
    assert answer.refused
    assert answer.text == "The model declined to answer this question. Not allowed."
    assert answer.cited_source_indices() == []


def test_detects_fallback_model():
    response = message(
        [
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-4-8"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": "Answer."},
        ],
        model="claude-opus-4-8",
    )
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES)
    assert answer.fallback_from == "claude-opus-5-5"
    assert answer.model == "claude-opus-4-8"
    assert answer.text == "Answer."


def test_max_tokens_marks_answer_truncated():
    response = message([{"type": "text", "text": "Cut"}], stop_reason="max_tokens")
    assert ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES).truncated
