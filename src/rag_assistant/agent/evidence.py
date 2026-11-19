"""What the agent has seen, and how many more tool calls it may make.

Evidence collects every source a tool showed the model, so that a citation in
the final answer can point back to it. ToolBudget caps how often each tool
runs: the limit lives in code, so the model can't talk its way past it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FetchedPage:
    """A page web_fetch read: its URL, its title if it has one, and its text (empty for a PDF)."""

    url: str
    title: str | None
    text: str


class Evidence:
    """The sources the answer can cite, in the order they arrived. Answer.sources is this list.

    Document passages and PyPI results are added when a tool returns them. Web pages
    are added when the answer cites them: web search and fetch run on Anthropic's
    servers, so their results reach us only inside the reply.
    """

    def __init__(self) -> None:
        self.sources: list[Source] = []
        self._index: dict[str, int] = {}  # key -> position in self.sources
        self._pages: list[FetchedPage] = []  # every page web_fetch read, in order

    def add(self, key: str, source: Source) -> int:
        """Register `source` under `key` and return its position. A repeated key keeps the first."""
        if key not in self._index:
            self._index[key] = len(self.sources)
            self.sources.append(source)
        return self._index[key]

    def record_fetches(self, message: BetaMessage) -> None:
        """Remember each page web_fetch read, to resolve citations of it later."""
        from anthropic.types.beta import (
            BetaPlainTextSource,
            BetaWebFetchBlock,
            BetaWebFetchToolResultBlock,
        )

        for block in message.content:
            if isinstance(block, BetaWebFetchToolResultBlock) and isinstance(
                block.content, BetaWebFetchBlock
            ):
                doc = block.content.content
                text = doc.source.data if isinstance(doc.source, BetaPlainTextSource) else ""
                self._pages.append(FetchedPage(block.content.url, doc.title, text))

    def _page(self, title: str | None, quote: str) -> FetchedPage | None:
        """The fetched page a citation names: the one with its title that contains its quote.

        A citation of a fetched page carries the page's title but not its URL, and
        titles repeat (or are missing), so the quoted text decides between pages.
        """
        same_title = [page for page in self._pages if page.title == title]
        quoted = [page for page in same_title if quote and quote in page.text]
        found = quoted or same_title  # a PDF's text isn't kept: its title has to do
        return found[0] if found else None

    def resolve(self, citation: BetaTextCitation) -> int | None:
        """The position in self.sources of what `citation` points to, or None."""
        from anthropic.types.beta import (
            BetaCitationSearchResultLocation,
            BetaCitationsWebSearchResultLocation,
        )

        if isinstance(citation, BetaCitationSearchResultLocation):
            # One of our tools returned it, under the key we gave it as `source`.
            return self._index.get(citation.source)
        # Web pages get keys of their own, so a web page never merges with the PyPI
        # result that has the same URL. The same page cited twice is one source.
        if isinstance(citation, BetaCitationsWebSearchResultLocation):
            title = citation.title or citation.url
            return self.add(f"web:{citation.url}", _web(citation.cited_text, citation.url, title))
        # The rest cite documents: here, only pages web_fetch read.
        page = self._page(citation.document_title, citation.cited_text)
        if page is None:  # not a page we saw fetched: there is nothing to point to
            return None
        title = page.title or page.url
        return self.add(f"web:{page.url}", _web(citation.cited_text, page.url, title))


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
