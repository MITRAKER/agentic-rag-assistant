"""What the agent has seen, and how many more tool calls it may make.

Evidence collects every source a tool showed the model, so that a citation in
the final answer can point back to it. ToolBudget caps how often each tool
runs: the limit lives in code, so the model can't talk its way past it.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from anthropic.lib.tools import ToolError

from rag_assistant.answering.types import Source

if TYPE_CHECKING:
    from anthropic.types.beta import BetaMessage, BetaTextCitation


def document_key(src: Source) -> str:
    """A unique, readable name for one chunk, e.g. 'docs/report.pdf#page=3&chunk=2'.

    The model sees it as the search result's `source`, and every citation of the
    chunk carries it back, which is how we find the chunk again.
    """
    parts = [f"page={src.metadata['page']}"] if "page" in src.metadata else []
    chunk = src.metadata.get("chunk_index")
    if chunk is None:  # not from our ingestion: fall back to a fingerprint of the text
        chunk = hashlib.sha256(src.text.encode()).hexdigest()[:12]
    parts.append(f"chunk={chunk}")
    return f"{src.source}#{'&'.join(parts)}"


class Evidence:
    """Every source the agent saw, in order. Answer.sources is this list."""

    def __init__(self) -> None:
        self.sources: list[Source] = []
        self._index: dict[str, int] = {}  # key -> position in self.sources
        self._fetched: dict[str, str] = {}  # web page title -> URL, from web_fetch results

    def add(self, key: str, source: Source) -> int:
        """Register `source` under `key` and return its position. A repeated key keeps the first."""
        if key not in self._index:
            self._index[key] = len(self.sources)
            self.sources.append(source)
        return self._index[key]

    def record_fetches(self, message: BetaMessage) -> None:
        """Remember the URL of each page web_fetch read, to resolve citations of it later."""
        from anthropic.types.beta import BetaWebFetchBlock, BetaWebFetchToolResultBlock

        for block in message.content:
            if isinstance(block, BetaWebFetchToolResultBlock) and isinstance(
                block.content, BetaWebFetchBlock
            ):
                title = block.content.content.title or block.content.url
                self._fetched.setdefault(title, block.content.url)

    def resolve(self, citation: BetaTextCitation) -> int | None:
        """The position in self.sources of what `citation` points to, or None."""
        from anthropic.types.beta import (
            BetaCitationSearchResultLocation,
            BetaCitationsWebSearchResultLocation,
        )

        if isinstance(citation, BetaCitationSearchResultLocation):
            # One of our tools returned it, under the key we gave it as `source`.
            return self._index.get(citation.source)
        if isinstance(citation, BetaCitationsWebSearchResultLocation):
            title = citation.title or citation.url
            return self.add(citation.url, _web(citation.cited_text, citation.url, title))
        # The rest cite documents: here, only pages web_fetch read. The citation names
        # the page by its title, so look the URL up by title.
        title = citation.document_title or "web page"
        url = self._fetched.get(title, title)
        return self.add(url, _web(citation.cited_text, url, title))


def _web(text: str, url: str, title: str) -> Source:
    return Source(text=text, source=url, title=title, score=0.0, kind="web")


class ToolBudget:
    """How many times each tool may run per attempt. The model is told when it runs out."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("the tool budget must be at least 1")
        self.limit = limit
        self._used: dict[str, int] = {}

    def spend(self, tool: str) -> None:
        used = self._used.get(tool, 0)
        if used >= self.limit:
            # ToolError becomes an error tool_result: the model reads this and
            # answers with what it has, instead of the loop crashing.
            raise ToolError(
                f"{tool} has reached its limit of {self.limit} uses for this question. "
                "Answer with the evidence you already have."
            )
        self._used[tool] = used + 1

    def reset(self) -> None:
        """A new attempt (a revision after the self-check) gets the full budget again."""
        self._used.clear()
