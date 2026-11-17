"""Chunking: split long documents into overlapping, search-sized pieces.

Why chunk at all?
  * An embedding squeezes a whole text into one vector. A 50-page PDF in one
    vector says "this is about many things" and matches nothing well.
  * The LLM's context window is limited, so we want to send only the few
    paragraphs that actually answer the question.

Strategy ("recursive character splitting"):
  1. Try to split on the biggest natural boundary first: blank lines
     (paragraphs). If a piece is still too big, split it on single newlines,
     then sentences, then words.
  2. Greedily pack the small pieces back together up to `chunk_size`.
  3. Start each new chunk with the last whole sentence(s) of the previous one,
     up to `overlap` characters, so context that straddles a boundary still
     appears whole somewhere. (Overlap snaps to sentence boundaries, or to
     word boundaries when there are none, so chunks never start mid-sentence.)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from rag_assistant.ingestion.loaders import Document, Metadata

SEPARATORS = ("\n\n", "\n", ". ", " ")
# Places where an overlap may start, best first: after a sentence end or a blank
# line, then after any line break, then after any whitespace (between words).
_BOUNDARIES = (
    re.compile(r"(?<=[.!?])\s+|\n\s*\n"),
    re.compile(r"\n"),
    re.compile(r"\s+"),
)


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: Metadata = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable id: the same file + position + text always gives the same id.

        This makes ingestion idempotent: re-running it overwrites chunks
        instead of creating duplicates.
        """
        key = f"{self.source}|{self.metadata.get('page', '')}|{self.chunk_index}|{self.text}"
        return hashlib.sha256(key.encode()).hexdigest()


def _split(text: str, chunk_size: int) -> list[str]:
    """Break text into pieces no longer than chunk_size, preferring big boundaries.

    Written as a loop over an explicit stack rather than as a recursive
    function (Power of Ten rule 1), with a provable bound on the number of
    steps (rule 2): each step handles one fragment, and each separator level
    can create at most len(text) + 1 fragments.
    """
    pieces: list[str] = []
    stack: list[tuple[str, int]] = [(text, 0)]  # (fragment, index of next separator to try)
    max_steps = (len(text) + 1) * (len(SEPARATORS) + 1) + 1
    for _ in range(max_steps):
        if not stack:
            return pieces
        fragment, level = stack.pop()
        if len(fragment) <= chunk_size:
            pieces.append(fragment)
        elif level == len(SEPARATORS):  # no boundary left: hard cut
            pieces.extend(fragment[i : i + chunk_size] for i in range(0, len(fragment), chunk_size))
        else:
            parts = _split_keeping_separator(fragment, SEPARATORS[level])
            # push in reverse so the first part is processed (and emitted) first
            stack.extend((part, level + 1) for part in reversed(parts))
    raise RuntimeError("chunking did not finish within its step bound")  # unreachable in practice


def _split_keeping_separator(text: str, sep: str) -> list[str]:
    """'a. b. c' split on '. ' -> ['a. ', 'b. ', 'c'], so joining gives the text back."""
    parts = text.split(sep)
    return [part + sep for part in parts[:-1]] + [parts[-1]]


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size")

    pieces = _split(text, chunk_size)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            # carry the tail of the previous chunk forward as overlap,
            # limited so the next chunk still has room for `piece`
            current = _overlap_tail(current, overlap, room=chunk_size - len(piece))
        current += piece
    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def _overlap_tail(chunk: str, overlap: int, room: int) -> str:
    """The text to repeat at the start of the next chunk.

    A raw `chunk[-overlap:]` usually starts mid-word or mid-sentence, and those
    fragments later show up in search results and citations. Instead we take
    the last whole sentences that fit in `overlap` characters. If even the last
    sentence is longer, we take that one sentence when it fits in `room` (the
    space left in the next chunk), and otherwise no overlap at all. Text with
    no sentence boundaries falls back to line breaks, then word breaks.
    """
    if overlap <= 0:
        return ""
    for boundary in _BOUNDARIES:
        ends = [m.end() for m in boundary.finditer(chunk)]
        if not ends:
            continue  # no boundary of this kind: try the next, finer kind
        # Possible starts: the chunk's own start (always clean) and every boundary inside it.
        starts = [0] + [e for e in ends if e < len(chunk)]
        fitting = [s for s in starts if len(chunk) - s <= overlap]
        start = fitting[0] if fitting else starts[-1]
        return chunk[start:] if len(chunk) - start <= room else ""
    return ""


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    return [
        Chunk(text=t, source=doc.source, chunk_index=i, metadata=dict(doc.metadata))
        for i, t in enumerate(split_text(doc.text, chunk_size, overlap))
    ]
