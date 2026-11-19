"""The query phase: retrieve evidence, then hand it to an answerer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from rag_assistant.answering.types import Answer, Answerer, Segment, Source

if TYPE_CHECKING:
    from rag_assistant.config import Settings

NO_RESULTS = (
    "I couldn't find anything relevant in your documents. "
    "Try rephrasing the question, or ingest documents that cover it."
)


def answer_question(
    question: str, retrieve: Callable[[str], list[Source]], answerer: Answerer
) -> Answer:
    """Find evidence with `retrieve` (usually a retrieval.Retriever), then answer from it."""
    sources = retrieve(question)
    if not sources:
        # Nothing to ground an answer in: say so instead of letting a model guess.
        # This also skips a paid API call.
        return Answer(question=question, segments=[Segment(NO_RESULTS)], sources=[])
    return answerer.answer(question, sources)


def get_answerer(settings: Settings) -> Answerer:
    if settings.llm == "extractive":
        from rag_assistant.answering.extractive import ExtractiveAnswerer

        return ExtractiveAnswerer()

    from rag_assistant.answering.claude import ClaudeAnswerer, client_from_settings

    return ClaudeAnswerer(
        client_from_settings(settings),
        model=settings.llm_model,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        fallbacks=settings.llm_fallbacks,
    )
