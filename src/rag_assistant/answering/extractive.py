"""Extractive answering: no LLM, just pick the sentences that best match.

This is the classic pre-LLM approach to question answering. It can't
rephrase, combine facts or say "no", but it is free, instant, deterministic
and works offline. We use it for tests, for demos without an API key, and in
step 5 as a baseline: a generative answerer has to beat it to be worth paying for.
"""

from __future__ import annotations

import re

from rag_assistant.answering.types import Answer, Citation, Segment, Source
from rag_assistant.text import split_sentences

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "you", "your",
}  # fmt: skip

NOT_FOUND = "The retrieved passages don't seem to answer this question."


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


class ExtractiveAnswerer:
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def answer(self, question: str, sources: list[Source]) -> Answer:
        wanted = keywords(question)
        candidates = []  # (overlap, -source_rank, -position, source_index, position, sentence)
        for i, src in enumerate(sources):
            for j, sentence in enumerate(split_sentences(src.text)):
                if sentence.startswith("#"):
                    continue  # a heading names a topic; it doesn't answer anything
                overlap = len(wanted & keywords(sentence))
                if overlap:
                    candidates.append((overlap, -i, -j, i, j, sentence))

        # Best overlap first; ties go to the better-ranked source, then earlier sentences.
        # Overlapping chunks can contain the same sentence twice, so skip repeats.
        best, seen = [], set()
        for candidate in sorted(candidates, reverse=True):
            if candidate[-1] not in seen and len(best) < self.max_sentences:
                seen.add(candidate[-1])
                best.append(candidate)
        if not best:
            return Answer(question, [Segment(NOT_FOUND)], sources, model="extractive")

        # Present the chosen sentences in reading order (source rank, then position).
        best.sort(key=lambda c: (c[3], c[4]))
        segments = [
            Segment(sentence + " ", [Citation(source_index=i, cited_text=sentence)])
            for *_, i, _j, sentence in best
        ]
        return Answer(question, segments, sources, model="extractive", stop_reason="end_turn")
