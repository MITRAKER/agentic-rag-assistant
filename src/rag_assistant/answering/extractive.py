"""Extractive answering: no LLM, just pick the sentences that best match.

This is the classic pre-LLM approach to question answering. It can't
rephrase, combine facts or say "no", but it is free, instant, deterministic
and works offline. We use it for tests, for demos without an API key, and in
step 5 as a baseline: a generative answerer has to beat it to be worth paying for.
"""

from __future__ import annotations

from typing import NamedTuple

from rag_assistant.answering.types import Answer, Citation, Segment, Source
from rag_assistant.text import keywords, split_sentences

NOT_FOUND = "The retrieved passages don't seem to answer this question."


class Candidate(NamedTuple):
    overlap: int  # question keywords found in the sentence
    source_index: int  # rank of the chunk it came from (0 = best match)
    position: int  # sentence number inside that chunk
    sentence: str

    def rank_key(self) -> tuple[int, int, int]:
        """Most keywords first; ties go to the better-ranked chunk, then earlier sentences."""
        return (-self.overlap, self.source_index, self.position)


class ExtractiveAnswerer:
    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be at least 1")
        self.max_sentences = max_sentences

    def answer(self, question: str, sources: list[Source]) -> Answer:
        best = self._pick(find_candidates(question, sources))
        if not best:
            return Answer(question, [Segment(NOT_FOUND)], sources, model="extractive")
        # Present the chosen sentences in reading order (chunk rank, then position).
        best.sort(key=lambda c: (c.source_index, c.position))
        segments = [
            Segment(
                c.sentence + " ", [Citation(source_index=c.source_index, cited_text=c.sentence)]
            )
            for c in best
        ]
        return Answer(question, segments, sources, model="extractive", stop_reason="end_turn")

    def _pick(self, candidates: list[Candidate]) -> list[Candidate]:
        """The best `max_sentences` candidates, skipping repeats.

        Overlapping chunks can contain the same sentence twice.
        """
        best: list[Candidate] = []
        seen: set[str] = set()
        for candidate in sorted(candidates, key=Candidate.rank_key):
            if candidate.sentence not in seen and len(best) < self.max_sentences:
                seen.add(candidate.sentence)
                best.append(candidate)
        return best


def find_candidates(question: str, sources: list[Source]) -> list[Candidate]:
    """Every sentence that shares at least one keyword with the question."""
    wanted = keywords(question)
    candidates: list[Candidate] = []
    for i, src in enumerate(sources):
        for j, sentence in enumerate(split_sentences(src.text)):
            if sentence.startswith("#"):
                continue  # a heading names a topic; it doesn't answer anything
            overlap = len(wanted & keywords(sentence))
            if overlap:
                candidates.append(Candidate(overlap, i, j, sentence))
    return candidates
