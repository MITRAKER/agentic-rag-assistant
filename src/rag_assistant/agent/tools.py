"""The agent's tools: where it can look for evidence.

* search_documents: the user's own documents (vector search plus reranking).
  A client tool: Claude asks for it, our code runs it.
* lookup_python_package: live data from an API, here PyPI. Also a client tool.
* web_search and web_fetch: server tools. Anthropic's servers run them, so
  they are only declared here.

A tool's description tells the model *when* to use it, not just what it does:
that text is how the model routes a question to a source.

Client tools return `search_result` blocks, the same blocks step 2 sends. A
citation of one copies the exact sentences it relies on, so the agent's answer
can only quote what a tool really returned.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx2
from anthropic import beta_tool
from anthropic.lib.tools import BetaFunctionTool, BetaFunctionToolResultType, ToolError

from rag_assistant import __version__
from rag_assistant.agent.evidence import Evidence, ToolBudget, document_key
from rag_assistant.answering.claude import search_result_block
from rag_assistant.answering.types import Source, Step

if TYPE_CHECKING:
    from anthropic.types.beta import BetaSearchResultBlockParam, BetaToolUnionParam

MAX_QUERY_CHARS = 500
PYPI_URL = "https://pypi.org/pypi/{name}/json"  # a fixed host: the model only picks the name
PYPI_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 5_000_000  # a big package's history is a few MB; refuse anything larger
# PEP 508 project names: letters, digits, ".", "_" and "-", starting and ending alphanumeric.
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]{0,98}[A-Za-z0-9])?")


def document_search_tool(
    retrieve: Callable[[str], list[Source]],
    evidence: Evidence,
    budget: ToolBudget,
    steps: list[Step],
) -> BetaFunctionTool[Any]:
    """The search_documents tool, bound to one question's evidence, budget and trace."""

    # eager_input_streaming: the requests are streamed, so let the tool input stream
    # too. @beta_tool validates the input against the signature before running it.
    @beta_tool(eager_input_streaming=True)
    def search_documents(query: str) -> BetaFunctionToolResultType:
        """Search the user's own documents: their notes, reports and other files.

        Use this first for every question, because the user's documents are the main
        source of truth. If the passages don't answer the question, search again with
        other words, a narrower query or a different angle. Returns the most relevant
        passages, best first.

        Args:
            query: What to look for, as a short search query, e.g. "HNSW index parameters".
        """
        query = _clean_query(query)
        budget.spend("search_documents")
        found = retrieve(query)
        steps.append(Step("search_documents", query, f"{len(found)} passage(s)"))
        if not found:
            return "No passages in the user's documents match this query."
        blocks: list[BetaSearchResultBlockParam] = []
        for src in found:
            key = document_key(src)
            evidence.add(key, src)
            blocks.append(search_result_block(src, source=key))
        return blocks

    return search_documents


def _clean_query(query: str) -> str:
    query = " ".join(query.split())
    if not query:
        raise ToolError("The query is empty. Write what to search for.")
    if len(query) > MAX_QUERY_CHARS:
        raise ToolError(f"The query is too long. Keep it under {MAX_QUERY_CHARS} characters.")
    return query


def package_lookup_tool(
    evidence: Evidence,
    budget: ToolBudget,
    steps: list[Step],
    transport: httpx2.BaseTransport | None = None,
) -> BetaFunctionTool[Any]:
    """The lookup_python_package tool. Tests pass a mock `transport` instead of the internet."""

    @beta_tool(eager_input_streaming=True)
    def lookup_python_package(name: str) -> BetaFunctionToolResultType:
        """Look up a Python package on PyPI, the Python Package Index.

        Returns live facts: the latest version, its release date, the package summary,
        the Python versions it supports and any known vulnerabilities. Use it for
        current facts about a Python library, such as "what is the latest version of
        qdrant-client?": documents and web pages may be out of date on these.

        Args:
            name: The package's name on PyPI, e.g. "qdrant-client".
        """
        if not _PACKAGE_NAME.fullmatch(name):
            raise ToolError(f"{name!r} isn't a valid PyPI package name.")
        budget.spend("lookup_python_package")
        data = fetch_package(name, transport)
        src = package_source(data)
        steps.append(Step("lookup_python_package", name, f"version {data['info']['version']}"))
        evidence.add(src.source, src)
        return [search_result_block(src, source=src.source)]

    return lookup_python_package


def fetch_package(name: str, transport: httpx2.BaseTransport | None = None) -> dict[str, Any]:
    """PyPI's JSON description of a package. Raises ToolError, which the model sees, on failure."""
    normalized = re.sub(r"[-_.]+", "-", name).lower()  # PEP 503 normalization
    url = PYPI_URL.format(name=normalized)
    headers = {"User-Agent": f"rag-assistant/{__version__}"}
    try:
        with (
            httpx2.Client(
                transport=transport, timeout=PYPI_TIMEOUT_SECONDS, headers=headers
            ) as http,
            http.stream("GET", url) as response,
        ):
            if response.status_code == httpx2.codes.NOT_FOUND:
                raise ToolError(f"PyPI has no package named {name!r}.")
            if response.status_code != httpx2.codes.OK:
                raise ToolError(f"PyPI answered with HTTP {response.status_code}. Try later.")
            body = _read_capped(response)
    except httpx2.HTTPError as e:
        raise ToolError(f"Couldn't reach PyPI: {e}") from e
    try:
        data = json.loads(body)
    except ValueError as e:
        raise ToolError("PyPI sent a reply that isn't valid JSON.") from e
    if not isinstance(data, dict) or not isinstance(data.get("info"), dict):
        raise ToolError("PyPI's reply has no package information.")
    return data


def _read_capped(response: httpx2.Response) -> bytes:
    """The response body, refusing to read more than MAX_RESPONSE_BYTES (Power of Ten rule 3)."""
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise ToolError("PyPI's reply is too large to read.")
        chunks.append(chunk)
    return b"".join(chunks)


def package_source(data: dict[str, Any]) -> Source:
    """Turn PyPI's JSON into a Source with one short sentence per fact (one citable block each)."""
    info = data["info"]
    name, version = str(info.get("name", "?")), str(info.get("version", "?"))
    facts = [f"The latest version of {name} on PyPI is {version}."]
    uploads = sorted(str(f.get("upload_time_iso_8601", "")) for f in data.get("urls") or [])
    if uploads and uploads[0]:
        facts.append(f"Version {version} was released on {uploads[0][:10]}.")
    summary = str(info.get("summary") or "").strip()
    if summary:
        end = "" if summary.endswith((".", "!", "?")) else "."  # no doubled full stop
        facts.append(f"PyPI describes it as: {summary}{end}")
    if info.get("requires_python"):
        facts.append(f"It requires Python {info['requires_python']}.")
    vulnerabilities = [str(v.get("id", "?")) for v in data.get("vulnerabilities") or []]
    if vulnerabilities:
        facts.append(
            f"PyPI lists {len(vulnerabilities)} known vulnerabilities for version {version}: "
            f"{', '.join(vulnerabilities[:5])}."
        )
    else:
        facts.append(f"PyPI lists no known vulnerabilities for version {version}.")
    url = f"https://pypi.org/project/{name}/"
    return Source("\n\n".join(facts), url, f"PyPI: {name}", score=0.0, kind="api")


def web_tools(max_uses: int) -> list[BetaToolUnionParam]:
    """Web search and web fetch. Anthropic runs them; the API bills each use."""
    return [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses},
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": max_uses,
            "max_content_tokens": 10_000,  # long pages are cut, to bound the cost
            "citations": {"enabled": True},
            # Fetch only URLs the user typed or web search found. A URL planted in a
            # document or in a fetched page can't be fetched, so text hidden there
            # can't send the agent, with your data in the URL, to a site of its choice.
            "url_sources": {
                "user_input": {"type": "all"},
                "client_tool_results": {"type": "none"},
                "server_tool_results": {
                    "type": "only",
                    "tools": [{"type": "tool_reference", "name": "web_search"}],
                },
            },
        },
    ]
