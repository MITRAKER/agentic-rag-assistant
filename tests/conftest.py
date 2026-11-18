import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.vectorstore import VectorStore, make_client


@pytest.fixture
def store() -> VectorStore:
    """An in-memory Qdrant with the offline hashing embedder: fast, no network."""
    settings = Settings(qdrant_path=":memory:", embedder="hashing", collection="test")
    return VectorStore(make_client(settings), settings.collection, HashingEmbedder())
