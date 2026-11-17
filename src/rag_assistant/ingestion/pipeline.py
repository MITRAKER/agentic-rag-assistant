"""The ingestion pipeline glues the steps together: load -> chunk -> store."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag_assistant.ingestion.chunking import Chunk, chunk_document
from rag_assistant.ingestion.loaders import load_path

if TYPE_CHECKING:  # imported only for type hints, avoids a circular import
    from rag_assistant.vectorstore import VectorStore


@dataclass
class IngestReport:
    files: int
    documents: int
    chunks: int


def ingest_path(
    path: str | Path, store: VectorStore, chunk_size: int = 800, overlap: int = 120
) -> IngestReport:
    docs = load_path(path)

    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for doc in docs:
        by_source[doc.source].extend(chunk_document(doc, chunk_size, overlap))

    total = 0
    for source, chunks in by_source.items():
        # Replace, don't append: if a file got shorter, old chunks must go.
        if store.client.collection_exists(store.collection):
            store.delete_source(source)
        total += store.upsert(chunks)

    return IngestReport(files=len(by_source), documents=len(docs), chunks=total)
