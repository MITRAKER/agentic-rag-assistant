import os

import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.vectorstore import VectorStore, make_client


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Every test sees only the settings it sets itself.

    Without this, a developer's own .env file (CONTRIBUTING suggests creating
    one) or exported RAG_* and ANTHROPIC_* variables would change what the
    tests run against: a Qdrant server instead of an in-memory one, or real
    Anthropic credentials where a test expects none.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name.startswith(("RAG_", "ANTHROPIC_")):
            monkeypatch.delenv(name)


@pytest.fixture
def store() -> VectorStore:
    """An in-memory Qdrant with the offline hashing embedder: fast, no network."""
    settings = Settings(qdrant_path=":memory:", embedder="hashing", collection="test")
    return VectorStore(make_client(settings), settings.collection, HashingEmbedder())
