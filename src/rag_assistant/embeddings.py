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
from typing import Protocol, cast

from rag_assistant.config import Settings
from rag_assistant.text import content_words


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingModelError(RuntimeError):
    """The embedding model couldn't be loaded: usually no internet on the first run."""


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        try:
            self._model = TextEmbedding(model_name)
        except ValueError as e:  # fastembed: "Could not load model ... from any source."
            raise EmbeddingModelError(
                f"Couldn't load the embedding model {model_name}: {e} The first run downloads "
                "it, so it needs an internet connection. To work offline, set "
                "RAG_EMBEDDER=hashing."
            ) from e
        self.dim = len(self.embed_query("probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # fastembed returns numpy arrays; Qdrant wants plain lists of floats.
        return [cast("list[float]", v.tolist()) for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models are trained with a different prefix for queries vs passages;
        # fastembed's query_embed adds it for us.
        return cast("list[float]", next(iter(self._model.query_embed(text))).tolist())


class HashingEmbedder:
    """Feature hashing: each content word adds 1 to one of `dim` buckets; L2-normalised.

    Two texts score high when they share words. Stopwords ("the", "what") are
    skipped because every text shares them. Different words can land in the
    same bucket ("collisions"), which adds a little noise but never hides a match.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim < 1:
            raise ValueError("dim must be at least 1")
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in content_words(text):
            # blake2b is used as a fast, well-spread hash here, not for security.
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % self.dim] += 1.0
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
