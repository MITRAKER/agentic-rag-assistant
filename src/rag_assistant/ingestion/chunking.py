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
  3. Start each new chunk with the last `overlap` characters of the previous
     one, so a sentence that straddles a boundary still appears whole somewhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from rag_assistant.ingestion.loaders import Document

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable id: the same file + position + text always gives the same id.

        This makes ingestion idempotent: re-running it overwrites chunks
        instead of creating duplicates.
        """
        key = f"{self.source}|{self.metadata.get('page', '')}|{self.chunk_index}|{self.text}"
        return hashlib.sha256(key.encode()).hexdigest()


def _split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than chunk_size, preferring big boundaries."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:  # no boundary left: hard cut
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split(text, chunk_size, rest)

    pieces: list[str] = []
    parts = text.split(sep)
    for i, part in enumerate(parts):
        # keep the separator attached so we can rebuild the text faithfully
        piece = part + sep if i < len(parts) - 1 else part
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split(piece, chunk_size, rest))
    return pieces


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    pieces = _split(text, chunk_size, SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            # carry the tail of the previous chunk forward as overlap,
            # trimmed so the next chunk still has room for `piece`
            tail_len = max(0, min(overlap, chunk_size - len(piece)))
            current = current[-tail_len:] if tail_len else ""
        current += piece
    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    return [
        Chunk(text=t, source=doc.source, chunk_index=i, metadata=dict(doc.metadata))
        for i, t in enumerate(split_text(doc.text, chunk_size, overlap))
    ]
