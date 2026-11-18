"""Data types for the query phase: what goes into an answerer and what comes out.

Keeping these as plain dataclasses (no LLM library types) means the CLI, the
API in step 4 and the evaluation in step 5 don't care which answerer ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from rag_assistant.ingestion.loaders import Metadata
from rag_assistant.vectorstore import SearchHit

SourceKind = Literal["document", "web", "api"]
Verdict = Literal["supported", "partly_supported", "unsupported", "no_answer", "skipped"]


@dataclass
class Source:
    """One piece of evidence: a retrieved chunk, a web page or an API result."""

    text: str
    source: str  # file path for documents, URL for web pages and API results
    title: str
    score: float  # cosine similarity to the question (documents only; 0.0 otherwise)
    metadata: Metadata = field(default_factory=dict)
    kind: SourceKind = "document"
    rerank_score: float | None = None  # set when a reranker ordered the chunks

    @classmethod
    def from_hit(cls, hit: SearchHit) -> Source:
        title = hit.metadata.get("title") or Path(hit.source).stem
        return cls(
            text=hit.text, source=hit.source, title=title, score=hit.score, metadata=hit.metadata
        )

    @property
    def label(self) -> str:
        """Human-readable location, e.g. 'handbook.pdf, page 4', or a URL."""
        if self.kind != "document":
            return self.source
        name = Path(self.source).name
        page = self.metadata.get("page")
        return f"{name}, page {page}" if page else name


@dataclass
class Citation:
    source_index: int  # position in Answer.sources (0-based)
    cited_text: str  # the exact passage from that source supporting the claim


@dataclass
class Segment:
    """A piece of the answer, plus the passages that back it (if any)."""

    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Step:
    """One action the agent took, kept so you can see how it reached its answer."""

    tool: str  # "search_documents", "web_search", "self_check", ...
    input: str  # the query, URL or package name it was given
    outcome: str  # what came back, e.g. "5 passages" or "error: ..."


@dataclass
class CheckResult:
    """The self-check's judgement of an answer against the passages it cites."""

    verdict: Verdict
    unsupported_claims: list[str] = field(default_factory=list)
    feedback: str = ""

    @property
    def passed(self) -> bool:
        """An honest "the sources don't cover this" passes too: it claims nothing unsupported."""
        return self.verdict in ("supported", "no_answer", "skipped")


@dataclass
class Answer:
    question: str
    segments: list[Segment]
    sources: list[Source]  # everything retrieved, in retrieval order
    model: str | None = None  # model that wrote the answer ("extractive" offline)
    stop_reason: str | None = None
    refused: bool = False
    fallback_from: str | None = None  # model that declined before a fallback answered
    input_tokens: int = 0
    output_tokens: int = 0
    steps: list[Step] = field(default_factory=list)  # the agent's actions, in order
    check: CheckResult | None = None  # set when the self-check ran

    @property
    def text(self) -> str:
        """The answer as plain text, without citation markers."""
        return "".join(s.text for s in self.segments).strip()

    @property
    def truncated(self) -> bool:
        """True when the model hit max_tokens, so the answer stops mid-way."""
        return self.stop_reason == "max_tokens"

    def cited_source_indices(self) -> list[int]:
        """Indices of cited sources, in the order they are first cited."""
        order: list[int] = []
        for seg in self.segments:
            for c in seg.citations:
                if c.source_index not in order:
                    order.append(c.source_index)
        return order

    def render(self) -> str:
        """The answer with [1], [2] markers after each cited piece.

        Sources are numbered by first appearance, so the reader sees [1], [2], [3]
        in order, whatever position each chunk had in the retrieval results.
        """
        number = {idx: n for n, idx in enumerate(self.cited_source_indices(), start=1)}
        parts = []
        for seg in self.segments:
            if not seg.citations:
                parts.append(seg.text)
                continue
            # Put the marker before trailing whitespace: "...claim. [1]\n\n"
            body = seg.text.rstrip()
            trailing = seg.text[len(body) :]
            nums = sorted({number[c.source_index] for c in seg.citations})
            parts.append(body + " " + "".join(f"[{n}]" for n in nums) + trailing)
        return "".join(parts).strip()

    @property
    def coverage(self) -> float:
        """Share of the answer's characters that carry a citation (0.0 to 1.0).

        A rough grounding signal: an answer that is 30% cited says a lot that
        isn't backed by your documents. The agent's self-check (step 3) goes
        further and asks whether each cited passage really supports its claim.
        """
        total = sum(len(s.text.strip()) for s in self.segments)
        cited = sum(len(s.text.strip()) for s in self.segments if s.citations)
        return cited / total if total else 0.0


class Answerer(Protocol):
    """Anything that turns a question + retrieved sources into an Answer."""

    def answer(self, question: str, sources: list[Source]) -> Answer:
        """An answer to `question` that cites `sources` by their position in the list."""
