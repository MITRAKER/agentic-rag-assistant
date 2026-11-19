"""Test doubles for the Anthropic client.

We never call the real API in tests: it costs money, needs a key and its
answers vary. Instead we check two contracts separately:
  1. the request we *send* has the right shape, and
  2. we *parse* a response of the documented shape correctly.
The responses are built with the SDK's own types, so a field we misspell
fails here rather than in production.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
from anthropic.types.beta import BetaMessage


def message(
    content: list[dict],
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    stop_details: dict | None = None,
    iterations: list[dict] | None = None,
) -> BetaMessage:
    usage: dict[str, Any] = {"input_tokens": 1200, "output_tokens": 80}
    if iterations is not None:
        usage["iterations"] = iterations
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage,
        }
    )


def iteration(kind: str, model: str) -> dict:
    """One `usage.iterations` entry: "message" for an attempt, "fallback_message" when a
    fallback model produced the response."""
    return {
        "type": kind,
        "model": model,
        "input_tokens": 1200,
        "output_tokens": 80,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def cited_text(text: str, index: int, quote: str) -> dict:
    """A text block citing search result `index`."""
    return {
        "type": "text",
        "text": text,
        "citations": [
            {
                "type": "search_result_location",
                "cited_text": quote,
                "source": f"doc{index}.md",
                "title": f"Doc {index}",
                "search_result_index": index,
                "start_block_index": 0,
                "end_block_index": 1,
            }
        ],
    }


class FakeStream:
    """What `client.beta.messages.stream(...)` returns: a context manager whose
    get_final_message() gives the finished response."""

    def __init__(self, response: BetaMessage):
        self.response = response

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def get_final_message(self) -> BetaMessage:
        return self.response


class FakeClient:
    """Mimics `anthropic.Anthropic()` just enough: client.beta.messages.stream(**kwargs)."""

    def __init__(self, response: BetaMessage):
        self.response = response
        self.requests: list[dict[str, Any]] = []
        self.beta = self
        self.messages = self

    def stream(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        return FakeStream(self.response)


# ---- The real SDK against a scripted API -------------------------------------------
# The agent runs inside the SDK's tool runner, which we can't replace with a fake
# client. Instead the real SDK talks to a mock HTTP transport that answers each
# request with the next reply from a script, as a stream of server-sent events.

Reply = str | httpx2.Response | Callable[[dict[str, Any]], str]


def sse_reply(
    message_id: str,
    blocks: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
) -> str:
    """One streamed reply holding `blocks`, in the event format the API uses."""
    start = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 1},
    }
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start}]
    for index, block in enumerate(blocks):
        events.extend(_block_events(index, block))
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 50},
        }
    )
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def _block_events(index: int, block: dict[str, Any]) -> list[dict[str, Any]]:
    """Text and tool calls arrive in pieces (deltas); tool results arrive whole."""
    kind = block["type"]
    if kind == "text":
        deltas = [{"type": "citations_delta", "citation": c} for c in block.get("citations", [])]
        deltas.append({"type": "text_delta", "text": block["text"]})
        first = {"type": "text", "text": ""}
    elif kind in ("tool_use", "server_tool_use"):
        deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        first = {**block, "input": {}}
    else:
        deltas, first = [], block
    return [
        {"type": "content_block_start", "index": index, "content_block": first},
        *({"type": "content_block_delta", "index": index, "delta": d} for d in deltas),
        {"type": "content_block_stop", "index": index},
    ]


def api_error(status: int, kind: str, message: str) -> httpx2.Response:
    """An error reply as the API sends it, e.g. api_error(529, "overloaded_error", "Overloaded")."""
    body = {"type": "error", "error": {"type": kind, "message": message}}
    return httpx2.Response(status, json=body)


def tool_call(name: str, tool_input: dict[str, Any], call_id: str = "toolu_1") -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": tool_input}


def text(words: str, *citations: dict[str, Any]) -> dict[str, Any]:
    return {"type": "text", "text": words, "citations": list(citations)}


def search_citation(source: str, quote: str, index: int = 0) -> dict[str, Any]:
    return {
        "type": "search_result_location",
        "cited_text": quote,
        "source": source,
        "title": "t",
        "search_result_index": index,
        "start_block_index": 0,
        "end_block_index": 1,
    }


class ScriptedAPI:
    """Plays the Anthropic API from a script of replies, with no network.

    A reply can be a function of the request, to answer with what it contains
    (for example, to cite a search result our own tool produced), or an error
    response (see api_error).
    """

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.betas: list[str] = []

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        if not self.replies:
            raise AssertionError("the client sent more requests than the script has replies")
        body = json.loads(request.content)
        self.requests.append(body)
        self.betas.append(request.headers.get("anthropic-beta", ""))
        reply = self.replies.pop(0)
        if isinstance(reply, httpx2.Response):
            return reply
        content = reply(body) if callable(reply) else reply
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, text=content)

    def client(self) -> anthropic.Anthropic:
        http = httpx2.Client(transport=httpx2.MockTransport(self._handle))
        return anthropic.Anthropic(api_key="sk-test", http_client=http, max_retries=0)
