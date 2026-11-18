"""Retrieval: vector search, then reranking.

Vector search uses a *bi-encoder*: the question and every chunk are embedded
separately, and chunks are ranked by how close their vectors are. That is fast
enough for millions of chunks, but coarse: one vector can't hold everything a
chunk says, and the question never "sees" the chunk it is compared with.

A *cross-encoder* reads the question and one passage together and scores how
well the passage answers it. It is much more accurate and much too slow to run
on every chunk. So retrieval runs in two stages, "retrieve wide, rerank narrow":

    question -> vector search returns RAG_RERANK_CANDIDATES chunks (20)
             -> the reranker scores each (question, chunk) pair
             -> keep the best RAG_TOP_K (5)

Three choices share the `Reranker` interface:
  * CrossEncoderReranker: a small local model (ms-marco MiniLM, about 80 MB, CPU).
  * LexicalReranker: no model. It fuses the vector ranking with a keyword
    ranking ("hybrid search"), which catches exact terms such as names, error
    codes and version numbers that embeddings blur. Works offline.
  * None: keep the vector search order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from rag_assistant.answering.types import Source
from rag_assistant.text import keywords

if TYPE_CHECKING:
    from rag_assistant.config import Settings
    from rag_assistant.vectorstore import VectorStore

RRF_K = 60  # the constant from the paper that introduced reciprocal rank fusion (Cormack, 2009)


class RerankerModelError(RuntimeError):
    """The reranking model couldn't be loaded: usually no internet on the first run."""


class Reranker(Protocol):
    def rerank(self, query: str, sources: list[Source], top_k: int) -> list[Source]:
        """The best `top_k` of `sources` for `query`, best first, each with its rerank_score."""


class CrossEncoderReranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        try:
            self._model = TextCrossEncoder(model_name)
        except ValueError as e:  # fastembed: "Could not load model ... from any source."
            raise RerankerModelError(
                f"Couldn't load the reranking model {model_name}: {e} The first run downloads "
                "it, so it needs an internet connection. To work offline, set "
                "RAG_RERANKER=lexical."
            ) from e

    def rerank(self, query: str, sources: list[Source], top_k: int) -> list[Source]:
        if not sources:
            return []
        # One score per passage; higher means more relevant. The scores are raw
        # model outputs ("logits"), useful for ordering but not probabilities.
        scores = [float(s) for s in self._model.rerank(query, [s.text for s in sources])]
        return best(sources, scores, top_k)


class LexicalReranker:
    """Hybrid ranking without a model: reciprocal rank fusion (RRF) of two rankings.

    The first ranking is the vector search order (sources arrive best first).
    The second orders the same chunks by how many of the question's keywords
    each one contains. A chunk earns 1 / (60 + rank) from each ranking, and the
    sums decide. RRF only uses positions, never raw scores, so it can combine a
    cosine similarity with a keyword count without calibrating one to the other.
    """

    def rerank(self, query: str, sources: list[Source], top_k: int) -> list[Source]:
        wanted = keywords(query)
        overlap = [len(wanted & keywords(s.text)) for s in sources]
        # sorted() is stable, so chunks with equal overlap keep their vector order.
        by_keywords = sorted(range(len(sources)), key=lambda i: -overlap[i])
        keyword_rank = {i: rank for rank, i in enumerate(by_keywords)}
        scores = [_rrf(i) + _rrf(keyword_rank[i]) for i in range(len(sources))]
        return best(sources, scores, top_k)


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)  # our ranks start at 0, the paper's at 1


def best(sources: list[Source], scores: list[float], top_k: int) -> list[Source]:
    """The `top_k` sources with the highest scores, best first, each carrying its score."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(scores) != len(sources):
        raise ValueError("need exactly one score per source")
    order = sorted(range(len(sources)), key=lambda i: -scores[i])  # ties keep vector order
    return [replace(sources[i], rerank_score=scores[i]) for i in order[:top_k]]


@dataclass(frozen=True)
class Retriever:
    """Vector search plus optional reranking: the one way the assistant finds evidence."""

    store: VectorStore
    top_k: int = 5
    min_score: float = 0.0
    reranker: Reranker | None = None
    candidates: int = 20  # how many chunks the reranker chooses from

    def __call__(self, query: str) -> list[Source]:
        if self.reranker is None:
            return self._search(query, self.top_k)
        # Fetch more than we keep, so the reranker can promote a chunk that vector
        # search placed lower down.
        found = self._search(query, max(self.candidates, self.top_k))
        return self.reranker.rerank(query, found, self.top_k) if found else []

    def _search(self, query: str, limit: int) -> list[Source]:
        hits = self.store.search(query, top_k=limit, min_score=self.min_score)
        return [Source.from_hit(h) for h in hits]


def get_reranker(settings: Settings) -> Reranker | None:
    choice = settings.reranker
    if choice == "auto":  # follow the embedder: a real model, or nothing to download
        choice = "cross-encoder" if settings.embedder == "fastembed" else "lexical"
    if choice == "cross-encoder":
        return CrossEncoderReranker(settings.rerank_model)
    if choice == "lexical":
        return LexicalReranker()
    return None


def make_retriever(
    settings: Settings, store: VectorStore, top_k: int | None = None, rerank: bool = True
) -> Retriever:
    return Retriever(
        store,
        top_k=top_k or settings.top_k,
        min_score=settings.min_score,
        reranker=get_reranker(settings) if rerank else None,
        candidates=settings.rerank_candidates,
    )
