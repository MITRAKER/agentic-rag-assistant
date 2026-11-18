"""The query phase: retrieve evidence, then hand it to an answerer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_assistant.answering.types import Answer, Answerer, Segment, Source

if TYPE_CHECKING:
    from rag_assistant.config import Settings
    from rag_assistant.vectorstore import VectorStore

NO_RESULTS = (
    "I couldn't find anything relevant in your documents. "
    "Try rephrasing the question, or ingest documents that cover it."
)


def retrieve(
    question: str, store: VectorStore, top_k: int = 5, min_score: float = 0.0
) -> list[Source]:
    return [Source.from_hit(h) for h in store.search(question, top_k=top_k, min_score=min_score)]


def answer_question(
    question: str,
    store: VectorStore,
    answerer: Answerer,
    top_k: int = 5,
    min_score: float = 0.0,
) -> Answer:
    sources = retrieve(question, store, top_k, min_score)
    if not sources:
        # Nothing to ground an answer in: say so instead of letting a model guess.
        # This also skips a paid API call.
        return Answer(question=question, segments=[Segment(NO_RESULTS)], sources=[])
    return answerer.answer(question, sources)


def get_answerer(settings: Settings) -> Answerer:
    if settings.llm == "extractive":
        from rag_assistant.answering.extractive import ExtractiveAnswerer

        return ExtractiveAnswerer()

    from rag_assistant.answering.claude import ClaudeAnswerer, make_client

    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    return ClaudeAnswerer(
        make_client(key or None),
        model=settings.llm_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        fallbacks=settings.llm_fallbacks,
    )
