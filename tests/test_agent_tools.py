"""The agent's tools, its evidence tracking and its self-check, one piece at a time."""

import json

import httpx2
import pytest
from anthropic.lib.tools import ToolError

from fakes import FakeClient, message, text
from rag_assistant.agent import SelfChecker
from rag_assistant.agent import tools as agent_tools
from rag_assistant.agent.check import check_prompt, parse_check
from rag_assistant.agent.evidence import Evidence, ToolBudget, document_key
from rag_assistant.agent.tools import document_search_tool, package_lookup_tool
from rag_assistant.answering import Answer, Citation, Segment, Source, Step

PYPI_JSON = {
    "info": {
        "name": "qdrant-client",
        "version": "1.20.0",
        "summary": "Client library for the Qdrant vector search engine",
        "requires_python": ">=3.10",
    },
    "urls": [
        {"upload_time_iso_8601": "2026-10-22T10:26:16Z"},
        {"upload_time_iso_8601": "2026-10-22T10:26:14Z"},
    ],
    "vulnerabilities": [],
}


def pypi_tool(handler, budget: int = 3):
    evidence, steps = Evidence(), []
    tool = package_lookup_tool(evidence, ToolBudget(budget), steps, httpx2.MockTransport(handler))
    return tool, evidence, steps


def test_package_lookup_returns_citable_facts():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx2.Response(200, json=PYPI_JSON)

    tool, evidence, steps = pypi_tool(handler)
    [block] = tool.call({"name": "Qdrant_Client"})

    assert urls == ["https://pypi.org/pypi/qdrant-client/json"]  # normalized name, fixed host
    assert block["source"] == "https://pypi.org/project/qdrant-client/"
    assert [b["text"] for b in block["content"]] == [
        "The latest version of qdrant-client on PyPI is 1.20.0.",
        "Version 1.20.0 was released on 2026-10-22.",
        "PyPI describes it as: Client library for the Qdrant vector search engine.",
        "It requires Python >=3.10.",
        "PyPI lists no known vulnerabilities for version 1.20.0.",
    ]
    assert evidence.sources[0].kind == "api"
    assert steps == [Step("lookup_python_package", "Qdrant_Client", "version 1.20.0")]


def test_package_lookup_reports_known_vulnerabilities():
    data = {**PYPI_JSON, "vulnerabilities": [{"id": "PYSEC-1"}, {"id": "GHSA-2"}], "urls": []}
    tool, _, _ = pypi_tool(lambda request: httpx2.Response(200, json=data))
    [block] = tool.call({"name": "qdrant-client"})
    sentences = [b["text"] for b in block["content"]]
    assert "PyPI lists 2 known vulnerabilities for version 1.20.0: PYSEC-1, GHSA-2." in sentences
    assert not any("released" in s for s in sentences)  # no files, no release date


def refuse_connection(request):
    raise httpx2.ConnectError("no network", request=request)


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (lambda r: httpx2.Response(404), "PyPI has no package named 'nope'"),
        (lambda r: httpx2.Response(503), "HTTP 503"),
        (lambda r: httpx2.Response(200, text="<html>"), "isn't valid JSON"),
        (lambda r: httpx2.Response(200, json=[1, 2]), "no package information"),
        (refuse_connection, "Couldn't reach PyPI"),
    ],
)
def test_package_lookup_failures_become_messages_for_the_model(handler, error):
    tool, _, steps = pypi_tool(handler)
    with pytest.raises(ToolError, match=error):
        tool.call({"name": "nope"})
    assert steps == []


def test_package_lookup_refuses_an_oversized_reply(monkeypatch):
    monkeypatch.setattr(agent_tools, "MAX_RESPONSE_BYTES", 100)
    tool, _, _ = pypi_tool(lambda r: httpx2.Response(200, content=b"x" * 101))
    with pytest.raises(ToolError, match="too large"):
        tool.call({"name": "big"})


def test_package_lookup_checks_the_name_before_any_request():
    def no_request(request):
        raise AssertionError("must not be sent")

    tool, _, _ = pypi_tool(no_request)
    for bad in ("../admin", "a b", "-leading", "x" * 101):
        with pytest.raises(ToolError, match="valid PyPI package name"):
            tool.call({"name": bad})


def test_package_lookup_has_a_budget():
    tool, _, _ = pypi_tool(lambda r: httpx2.Response(200, json=PYPI_JSON), budget=1)
    tool.call({"name": "qdrant-client"})
    with pytest.raises(ToolError, match="limit of 1 uses"):
        tool.call({"name": "qdrant-client"})


def test_document_search_validates_the_query():
    tool = document_search_tool(lambda q: [], Evidence(), ToolBudget(5), [])
    with pytest.raises(ToolError, match="empty"):
        tool.call({"query": "   "})
    with pytest.raises(ToolError, match="too long"):
        tool.call({"query": "x" * 501})
    assert tool.call({"query": "  spaced   out  "}) == (
        "No passages in the user's documents match this query."
    )


def test_document_search_registers_each_chunk_once():
    chunk = Source("Qdrant uses HNSW.", "docs/v.md", "V", 0.9, {"chunk_index": 1})
    evidence, steps = Evidence(), []
    tool = document_search_tool(lambda q: [chunk], evidence, ToolBudget(5), steps)
    [first] = tool.call({"query": "hnsw"})
    tool.call({"query": "hnsw again"})
    assert first["source"] == "docs/v.md#chunk=1"
    assert evidence.sources == [chunk]
    assert [s.outcome for s in steps] == ["1 passage(s)", "1 passage(s)"]


def test_document_key_is_unique_per_page_and_chunk():
    pdf = Source("t", "r.pdf", "R", 0.5, {"page": 3, "chunk_index": 2})
    assert document_key(pdf) == "r.pdf#page=3&chunk=2"
    foreign = Source("some text", "x.md", "X", 0.5)  # not from our ingestion: no chunk_index
    assert document_key(foreign).startswith("x.md#chunk=")
    assert document_key(foreign) != document_key(Source("other text", "x.md", "X", 0.5))


def test_evidence_drops_citations_of_unknown_results():
    response = message(
        [
            text(
                "Claim.",
                {
                    "type": "search_result_location",
                    "cited_text": "Claim.",
                    "source": "never-returned.md#chunk=0",
                    "title": "t",
                    "search_result_index": 0,
                    "start_block_index": 0,
                    "end_block_index": 1,
                },
            )
        ]
    )
    [citation] = response.content[0].citations
    assert Evidence().resolve(citation) is None


def test_tool_budget_rejects_zero_and_resets():
    with pytest.raises(ValueError, match="at least 1"):
        ToolBudget(0)
    budget = ToolBudget(1)
    budget.spend("t")
    budget.reset()
    budget.spend("t")  # allowed again after a reset


# ---- self-check ---------------------------------------------------------------------

SOURCES = [Source("Qdrant uses HNSW.", "docs/v.md", "V", 0.9)]


def checked_answer() -> Answer:
    return Answer(
        "What index?",
        [
            Segment("Qdrant uses HNSW.", [Citation(0, "Qdrant uses HNSW.")]),
            Segment("\n\n"),
            Segment("It is the fastest index."),
        ],
        SOURCES,
    )


def test_check_prompt_pairs_each_statement_with_its_passages():
    prompt = check_prompt(checked_answer())
    assert "<question>\nWhat index?\n</question>" in prompt
    assert '[1] Qdrant uses HNSW.\nCited passage from v.md: "Qdrant uses HNSW."' in prompt
    assert "[2] It is the fastest index.\nCited passages: none" in prompt  # blank segment skipped


def test_checker_request_asks_for_json_in_a_fixed_shape():
    client = FakeClient(
        message(
            [text(json.dumps({"verdict": "supported", "unsupported_claims": [], "feedback": ""}))]
        )
    )
    result = SelfChecker(client).check(checked_answer())
    assert result.passed
    [request] = client.requests
    schema = request["output_config"]["format"]["schema"]
    assert schema["required"] == ["verdict", "unsupported_claims", "feedback"]
    assert schema["additionalProperties"] is False


def test_checker_skips_answers_that_are_incomplete():
    client = FakeClient(message([text("unused")]))
    checker = SelfChecker(client)
    refused = Answer("q", [Segment("No.")], [], refused=True)
    truncated = Answer("q", [Segment("Half")], [], stop_reason="max_tokens")
    for answer in (refused, truncated, Answer("q", [], [])):
        assert checker.check(answer).verdict == "skipped"
    assert client.requests == []
    with pytest.raises(ValueError, match="max_tokens"):
        SelfChecker(client, max_tokens=0)


@pytest.mark.parametrize(
    ("response", "verdict"),
    [
        (
            message(
                [text('{"verdict": "unsupported", "unsupported_claims": ["x"], "feedback": "f"}')]
            ),
            "unsupported",
        ),
        (message([text("not json")]), "skipped"),
        (message([text('{"verdict": "great"}')]), "skipped"),
        (message([text("")], stop_reason="refusal"), "skipped"),
        (message([text('{"verdict": "supp')], stop_reason="max_tokens"), "skipped"),
    ],
)
def test_parse_check_never_trusts_a_malformed_reply(response, verdict):
    assert parse_check(response).verdict == verdict
