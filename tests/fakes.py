"""Test doubles for the Anthropic client.

We never call the real API in tests: it costs money, needs a key and its
answers vary. Instead we check two contracts separately:
  1. the request we *send* has the right shape, and
  2. we *parse* a response of the documented shape correctly.
The responses are built with the SDK's own types, so a field we misspell
fails here rather than in production.
"""

from __future__ import annotations

from typing import Any

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
