"""Thin wrapper around Qdrant, the vector database.

Qdrant stores, for each chunk: an id, its vector, and a JSON "payload"
(the text, source file, page...). A search embeds the question and asks
Qdrant for the stored vectors with the highest cosine similarity.

Qdrant runs in two modes with the same API:
  * embedded: QdrantClient(path="./qdrant_data") - a folder on disk, no server
  * server:   QdrantClient(url="http://localhost:6333") - the Docker image
"""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from rag_assistant.config import Settings
from rag_assistant.embeddings import Embedder
from rag_assistant.ingestion.chunking import Chunk
from rag_assistant.ingestion.loaders import Metadata


@dataclass
class SearchHit:
    text: str
    source: str
    score: float
    metadata: Metadata


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    if settings.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=settings.qdrant_path)


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder):
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE
            ),
        )
        # An index on "source" makes filtering/deleting by file fast on a Qdrant
        # server. Embedded mode ignores it (and warns), which is fine for small data.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.client.create_payload_index(
                self.collection, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD
            )

    def delete_source(self, source: str) -> None:
        """Remove every chunk of one file, so a re-ingest leaves no stale chunks."""
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="source", match=models.MatchValue(value=source))
                    ]
                )
            ),
        )

    def upsert(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.ensure_collection()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            points = [
                models.PointStruct(
                    # Qdrant ids must be ints or UUIDs; derive a UUID from our hash.
                    id=str(uuid.UUID(c.id[:32])),
                    vector=vec,
                    payload={
                        "text": c.text,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        **c.metadata,
                    },
                )
                for c, vec in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(self.collection, points=points)
        return len(chunks)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[SearchHit]:
        """Return the `top_k` most similar chunks, best first.

        `min_score` drops weak matches (cosine similarity below it). Nearest
        neighbour search always returns *something*, even for a question your
        documents know nothing about, so a threshold is a cheap first filter.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.client.collection_exists(self.collection):
            return []
        result = self.client.query_points(
            self.collection,
            query=self.embedder.embed_query(query),
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
            with_payload=True,
        )
        hits = []
        for p in result.points:
            payload = dict(p.payload or {})
            hits.append(
                SearchHit(
                    text=payload.pop("text", ""),
                    source=payload.pop("source", ""),
                    score=p.score,
                    metadata=payload,
                )
            )
        return hits

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count
