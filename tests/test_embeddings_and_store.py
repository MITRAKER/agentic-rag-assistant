import pytest

from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.ingestion.chunking import Chunk


def test_hashing_embedder_is_normalised_and_ignores_stopwords():
    emb = HashingEmbedder(dim=64)
    [vec] = emb.embed_documents(["vector vector database"])
    assert len(vec) == 64
    assert sum(v * v for v in vec) == pytest.approx(1.0)
    assert emb.embed_query("the what of") == [0.0] * 64  # only stopwords: empty vector


def test_hashing_embedder_rejects_bad_dim():
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(dim=0)


def test_search_rejects_bad_top_k(store):
    with pytest.raises(ValueError, match="top_k"):
        store.search("q", top_k=0)


def test_upsert_rejects_bad_batch_size(store):
    with pytest.raises(ValueError, match="batch_size"):
        store.upsert([Chunk("t", "s", 0)], batch_size=0)
