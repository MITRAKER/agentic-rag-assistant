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
) -> BetaMessage:
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
            "usage": {"input_tokens": 1200, "output_tokens": 80},
        }
    )


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


class FakeClient:
    """Mimics `anthropic.Anthropic()` just enough: client.beta.messages.create(**kwargs)."""

    def __init__(self, response: BetaMessage):
        self.response = response
        self.requests: list[dict[str, Any]] = []
        self.beta = self
        self.messages = self

    def create(self, **kwargs: Any) -> BetaMessage:
        self.requests.append(kwargs)
        return self.response
