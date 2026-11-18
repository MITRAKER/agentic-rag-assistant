import json

import anthropic
import httpx2
import pytest

from fakes import FakeClient, cited_text, iteration, message
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
    assert not answer.refused
    assert not answer.truncated


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


def test_detects_a_sticky_fallback_without_a_fallback_block():
    """After a fallback, the API sends the same conversation straight to the fallback
    model for a while. Nothing declines, so there is no fallback block; the usage
    entry of type "fallback_message" is the signal."""
    response = message(
        [{"type": "text", "text": "Answer."}],
        model="claude-opus-4-8",
        iterations=[iteration("fallback_message", "claude-opus-4-8")],
    )
    answer = ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES)
    assert answer.fallback_from == "claude-opus-5-5"  # the model we asked for
    assert answer.model == "claude-opus-4-8"


def test_no_fallback_reported_when_the_requested_model_answered():
    response = message(
        [{"type": "text", "text": "Answer."}],
        iterations=[iteration("message", "claude-opus-5-5")],
    )
    assert ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES).fallback_from is None


def sse(*events: dict) -> str:
    """Server-sent events, the format the Messages API streams in."""
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


CITATION = {
    "type": "search_result_location",
    "cited_text": "Qdrant uses HNSW.",
    "source": "docs/vector.md",
    "title": "Vector Databases",
    "search_result_index": 0,
    "start_block_index": 0,
    "end_block_index": 1,
}
STREAM = sse(
    {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "citations_delta", "citation": CITATION},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Qdrant uses an HNSW index."},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 42},
    },
    {"type": "message_stop"},
)


def test_streams_through_the_real_sdk_with_a_large_max_tokens():
    """The real SDK, still offline: a mock transport plays the API's side.

    Without streaming, the SDK refuses any max_tokens above 21,333 before sending
    anything, so this also checks the top of RAG_LLM_MAX_TOKENS's range.
    """
    sent: list[httpx2.Request] = []

    def fake_api(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=STREAM)

    http = httpx2.Client(transport=httpx2.MockTransport(fake_api))
    with anthropic.Anthropic(api_key="sk-test", http_client=http, max_retries=0) as client:
        answer = ClaudeAnswerer(client, max_tokens=128_000).answer("What does Qdrant use?", SOURCES)

    [request] = sent
    body = json.loads(request.content)
    assert body["stream"] is True
    assert body["max_tokens"] == 128_000
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert answer.render() == "Qdrant uses an HNSW index. [1]"
    assert answer.segments[0].citations[0].cited_text == "Qdrant uses HNSW."
    assert (answer.input_tokens, answer.output_tokens) == (1200, 42)


def test_max_tokens_marks_answer_truncated():
    response = message([{"type": "text", "text": "Cut"}], stop_reason="max_tokens")
    assert ClaudeAnswerer(FakeClient(response)).answer("q", SOURCES).truncated


def test_rejects_empty_sources_and_bad_max_tokens():
    client = FakeClient(message([{"type": "text", "text": "ok"}]))
    with pytest.raises(ValueError, match="at least one source"):
        ClaudeAnswerer(client).answer("q", [])
    with pytest.raises(ValueError, match="max_tokens"):
        ClaudeAnswerer(client, max_tokens=0)
