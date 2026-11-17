"""The agent, end to end, through the real SDK and its tool runner, against a scripted API."""

import json
from pathlib import Path

import pytest

from fakes import (
    Reply,
    ScriptedAPI,
    message,
    search_citation,
    sse_reply,
    text,
    tool_call,
)
from rag_assistant.agent import AgentConfig, ClaudeAgent, SelfChecker, get_agent
from rag_assistant.agent.loop import revision_request, server_tool_steps, system_prompt
from rag_assistant.agent.tools import web_tools
from rag_assistant.answering import CheckResult, Step
from rag_assistant.config import Settings
from rag_assistant.ingestion import ingest_path
from rag_assistant.retrieval import LexicalReranker, Retriever

SAMPLE = Path(__file__).parent.parent / "data" / "sample"
OFFLINE = AgentConfig(web=False, pypi=False)  # just the document search tool


def answer_citing_the_first_result(words: str, message_id: str = "msg_2") -> Reply:
    """A final reply that cites the first passage our search tool returned."""

    def reply(body):
        [tool_result] = body["messages"][-1]["content"]
        first = tool_result["content"][0]
        quote = first["content"][0]["text"]
        return sse_reply(message_id, [text(words, search_citation(first["source"], quote))])

    return reply


def check_reply(verdict: str, claims: tuple[str, ...] = (), feedback: str = "") -> str:
    payload = {"verdict": verdict, "unsupported_claims": list(claims), "feedback": feedback}
    return sse_reply("msg_check", [text(json.dumps(payload))], model="claude-haiku-5-5")


def test_agent_searches_the_documents_and_cites_them(store):
    ingest_path(SAMPLE, store)
    api = ScriptedAPI(
        sse_reply("msg_1", [tool_call("search_documents", {"query": "HNSW index"})], "tool_use"),
        answer_citing_the_first_result("Qdrant uses HNSW."),
    )
    retriever = Retriever(store, top_k=2, reranker=LexicalReranker())
    answer = ClaudeAgent(api.client(), retriever).answer("What index does Qdrant use?")

    assert answer.render() == "Qdrant uses HNSW. [1]"
    [cited] = [answer.sources[i] for i in answer.cited_source_indices()]
    assert cited.source.endswith("vector-databases.md")
    assert cited.kind == "document"
    assert answer.steps == [Step("search_documents", "HNSW index", "2 passage(s)")]
    assert (answer.input_tokens, answer.output_tokens) == (2000, 100)

    first, second = api.requests
    assert first["thread"] == {"type": "create"}
    assert [t["name"] for t in first["tools"]] == [
        "search_documents",
        "lookup_python_package",
        "web_search",
        "web_fetch",
    ]
    assert first["tools"][0]["eager_input_streaming"] is True
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"] == {"effort": "medium"}
    assert first["fallbacks"] == "default"
    # The second request continues the thread: it sends only the tool's result.
    assert second["thread"] == {"type": "continue", "previous_message_id": "msg_1"}
    [user_turn] = second["messages"]
    [result] = user_turn["content"]
    assert result["type"] == "tool_result"
    assert {r["type"] for r in result["content"]} == {"search_result"}
    assert all(r["citations"] == {"enabled": True} for r in result["content"])


def test_disabled_tools_are_neither_sent_nor_mentioned(store):
    api = ScriptedAPI(sse_reply("msg_1", [text("The documents don't cover this.")]))
    config = AgentConfig(web=False, pypi=False, fallbacks=False)
    answer = ClaudeAgent(api.client(), Retriever(store), config).answer("Who won in 1998?")

    [request] = api.requests
    assert [t["name"] for t in request["tools"]] == ["search_documents"]
    assert "web_search" not in request["system"][0]["text"]
    assert "fallbacks" not in request
    assert answer.text == "The documents don't cover this."
    assert answer.check is None  # no checker given


def test_the_tool_budget_is_enforced_in_code(store):
    ingest_path(SAMPLE, store)
    api = ScriptedAPI(
        sse_reply("msg_1", [tool_call("search_documents", {"query": "HNSW"})], "tool_use"),
        sse_reply(
            "msg_2", [tool_call("search_documents", {"query": "graph", "x": 1}, "t2")], "tool_use"
        ),
        sse_reply("msg_3", [tool_call("search_documents", {"query": "graph"}, "t3")], "tool_use"),
        sse_reply("msg_4", [text("Qdrant uses HNSW.")]),
    )
    config = AgentConfig(tool_budget=1, web=False, pypi=False)
    answer = ClaudeAgent(api.client(), Retriever(store), config).answer("q")

    # An unexpected argument fails validation; the tool never runs.
    [bad_input] = api.requests[2]["messages"][0]["content"]
    assert bad_input["is_error"] is True
    # The second real search is over the budget of one: the model is told so.
    [over_budget] = api.requests[3]["messages"][0]["content"]
    assert over_budget["is_error"] is True
    assert "limit of 1 uses" in over_budget["content"]
    assert [s.input for s in answer.steps] == ["HNSW"]
    assert answer.text == "Qdrant uses HNSW."


def test_max_turns_ends_a_loop_that_would_go_on(store):
    api = ScriptedAPI(
        sse_reply("msg_1", [tool_call("search_documents", {"query": "a"})], "tool_use"),
        sse_reply("msg_2", [tool_call("search_documents", {"query": "b"}, "t2")], "tool_use"),
    )
    config = AgentConfig(max_turns=2, web=False, pypi=False)
    answer = ClaudeAgent(api.client(), Retriever(store), config).answer("q")

    assert len(api.requests) == 2  # never a third
    assert answer.stop_reason == "tool_use"
    assert answer.text == ""
    assert [s.outcome for s in answer.steps] == ["0 passage(s)", "0 passage(s)"]


WEB_SEARCH = {
    "type": "server_tool_use",
    "id": "srvtoolu_1",
    "name": "web_search",
    "input": {"query": "qdrant hnsw"},
}
WEB_RESULTS = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_1",
    "content": [
        {
            "type": "web_search_result",
            "url": "https://qdrant.tech/hnsw",
            "title": "HNSW in Qdrant",
            "encrypted_content": "abc",
            "page_age": None,
        }
    ],
}
WEB_CITATION = {
    "type": "web_search_result_location",
    "url": "https://qdrant.tech/hnsw",
    "title": "HNSW in Qdrant",
    "encrypted_index": "xyz",
    "cited_text": "Qdrant builds an HNSW graph.",
}


def test_web_search_results_become_cited_web_sources(store):
    reply = [
        text("Let me search the web."),
        WEB_SEARCH,
        WEB_RESULTS,
        text("Qdrant builds an HNSW graph.", WEB_CITATION),
    ]
    api = ScriptedAPI(sse_reply("msg_1", reply))
    answer = ClaudeAgent(api.client(), Retriever(store), AgentConfig(pypi=False)).answer("q")

    # The line written before the search isn't part of the answer.
    assert answer.render() == "Qdrant builds an HNSW graph. [1]"
    [page] = answer.sources
    assert (page.kind, page.source, page.title) == (
        "web",
        "https://qdrant.tech/hnsw",
        "HNSW in Qdrant",
    )
    assert page.label == "https://qdrant.tech/hnsw"
    assert answer.steps == [Step("web_search", "qdrant hnsw", "1 result(s)")]


def test_web_fetch_citations_point_to_the_page_url(store):
    fetch = {
        "type": "server_tool_use",
        "id": "srvtoolu_2",
        "name": "web_fetch",
        "input": {"url": "https://example.com/hnsw"},
    }
    fetched = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srvtoolu_2",
        "content": {
            "type": "web_fetch_result",
            "url": "https://example.com/hnsw",
            "retrieved_at": "2026-11-17T10:00:00Z",
            "content": {
                "type": "document",
                "title": "HNSW explained",
                "source": {"type": "text", "media_type": "text/plain", "data": "HNSW is a graph."},
            },
        },
    }
    cite = {
        "type": "char_location",
        "cited_text": "HNSW is a graph.",
        "document_index": 0,
        "document_title": "HNSW explained",
        "start_char_index": 0,
        "end_char_index": 16,
    }
    api = ScriptedAPI(sse_reply("msg_1", [fetch, fetched, text("HNSW is a graph.", cite)]))
    answer = ClaudeAgent(api.client(), Retriever(store)).answer("Summarize example.com/hnsw")

    [page] = answer.sources
    assert (page.source, page.title, page.kind) == (
        "https://example.com/hnsw",
        "HNSW explained",
        "web",
    )
    assert answer.steps == [Step("web_fetch", "https://example.com/hnsw", "page read")]


def test_server_tool_errors_appear_in_the_trace():
    reply = message(
        [
            WEB_SEARCH,
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": {
                    "type": "web_search_tool_result_error",
                    "error_code": "max_uses_exceeded",
                },
            },
            {
                "type": "server_tool_use",
                "id": "srvtoolu_2",
                "name": "web_fetch",
                "input": {"url": "u"},
            },
            {
                "type": "web_fetch_tool_result",
                "tool_use_id": "srvtoolu_2",
                "content": {"type": "web_fetch_tool_result_error", "error_code": "url_not_allowed"},
            },
        ]
    )
    assert server_tool_steps(reply) == [
        Step("web_search", "qdrant hnsw", "error: max_uses_exceeded"),
        Step("web_fetch", "u", "error: url_not_allowed"),
    ]


def test_a_failed_check_leads_to_one_revision_on_the_same_thread(store):
    claim = "Qdrant was founded in 2021."
    api = ScriptedAPI(
        sse_reply("msg_1", [text(claim)]),
        check_reply("unsupported", (claim,), "Cite a source for the date or remove it."),
        sse_reply("msg_2", [text("The documents don't say when Qdrant was founded.")]),
        check_reply("no_answer"),
    )
    client = api.client()
    agent = ClaudeAgent(client, Retriever(store), OFFLINE, SelfChecker(client))
    answer = agent.answer("When was Qdrant founded?")

    assert answer.text == "The documents don't say when Qdrant was founded."
    assert answer.check == CheckResult("no_answer", [], "")
    assert [(s.tool, s.outcome) for s in answer.steps] == [
        ("self_check", "unsupported"),
        ("self_check", "no_answer"),
    ]
    check = api.requests[1]
    assert check["model"] == "claude-haiku-5-5"
    assert check["output_config"]["effort"] == "low"
    assert check["output_config"]["format"]["type"] == "json_schema"
    assert "thread" not in check  # a separate, one-off request
    assert claim in check["messages"][0]["content"]
    revision = api.requests[2]
    assert revision["thread"] == {"type": "continue", "previous_message_id": "msg_1"}
    [request_for_revision] = revision["messages"]
    assert f"- {claim}" in request_for_revision["content"]
    assert "Write the whole answer again" in request_for_revision["content"]


def test_without_retries_a_failed_check_is_reported_not_retried(store):
    api = ScriptedAPI(
        sse_reply("msg_1", [text("Made up.")]),
        check_reply("unsupported", ("Made up.",), "Remove it."),
    )
    client = api.client()
    config = AgentConfig(web=False, pypi=False, retries=0)
    answer = ClaudeAgent(client, Retriever(store), config, SelfChecker(client)).answer("q")
    assert answer.check is not None
    assert not answer.check.passed
    assert len(api.requests) == 2


def test_a_refusal_is_reported_and_not_checked(store):
    api = ScriptedAPI(sse_reply("msg_1", [text("I can't")], "refusal"))
    client = api.client()
    answer = ClaudeAgent(client, Retriever(store), OFFLINE, SelfChecker(client)).answer("q")
    assert answer.refused
    assert answer.check == CheckResult(
        "skipped", feedback="Nothing to check: the answer is incomplete."
    )
    assert len(api.requests) == 1


def test_web_fetch_only_opens_urls_from_the_user_or_web_search():
    search, fetch = web_tools(3)
    assert search == {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}
    assert fetch["citations"] == {"enabled": True}
    sources = fetch["url_sources"]
    assert sources["client_tool_results"] == {"type": "none"}  # not from our documents
    assert sources["server_tool_results"]["tools"] == [
        {"type": "tool_reference", "name": "web_search"}  # not from fetched pages
    ]


def test_system_prompt_names_only_the_enabled_tools():
    full = system_prompt(AgentConfig())
    assert "web_search" in full
    assert "lookup_python_package" in full
    bare = system_prompt(OFFLINE)
    assert "search_documents" in bare
    assert "web_search" not in bare
    assert "lookup_python_package" not in bare


def test_revision_request_lists_claims_and_feedback():
    check = CheckResult("partly_supported", [f"claim {i}" for i in range(12)], "Fix the dates.")
    request = revision_request(check)
    assert "partly_supported" in request
    assert "- claim 9" in request
    assert "claim 10" not in request  # at most ten are listed
    assert "Reviewer's note: Fix the dates." in request
    assert revision_request(CheckResult("unsupported")).count("\n") == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_turns", 0), ("tool_budget", 0), ("max_tokens", 0), ("retries", -1)],
)
def test_agent_config_rejects_impossible_limits(field, value):
    with pytest.raises(ValueError, match=field):
        AgentConfig(**{field: value})


def test_get_agent_builds_the_agent_from_settings(store):
    settings = Settings(
        anthropic_api_key="sk-test",
        agent_max_turns=5,
        agent_tool_budget=2,
        agent_web=False,
        agent_retries=0,
        agent_check_model="claude-sonnet-5-5",
    )
    agent = get_agent(settings, Retriever(store))
    assert (agent.config.max_turns, agent.config.tool_budget) == (5, 2)
    assert (agent.config.web, agent.config.pypi, agent.config.retries) == (False, True, 0)
    assert agent.checker is not None
    assert agent.checker.model == "claude-sonnet-5-5"
    no_check = get_agent(
        Settings(anthropic_api_key="sk-test", agent_self_check=False), Retriever(store)
    )
    assert no_check.checker is None
