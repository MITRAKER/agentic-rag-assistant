"""Embeddings: turn text into vectors so that similar meaning = nearby vectors.

Two implementations share one tiny interface (`Embedder`):

* FastEmbedEmbedder: a real semantic model (BAAI/bge-small-en-v1.5, 384 dims)
  running locally on CPU via ONNX. No API key, no cost. Downloads once.
* HashingEmbedder: a deterministic bag-of-words hash. It understands no
  meaning, but it needs no download, so tests and offline demos still work.

Because the rest of the code only talks to the `Embedder` protocol, swapping
in OpenAI, Voyage or Cohere embeddings later is a one-class change.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from rag_assistant.config import Settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models are trained with a different prefix for queries vs passages;
        # fastembed's query_embed adds it for us.
        return next(iter(self._model.query_embed(text))).tolist()


class HashingEmbedder:
    """Feature hashing: each word bumps one of `dim` buckets. L2-normalised."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    return FastEmbedEmbedder(settings.embedding_model)
