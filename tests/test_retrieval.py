import sys
from pathlib import Path
from types import ModuleType

import pytest

from rag_assistant.answering import Source
from rag_assistant.config import Settings
from rag_assistant.ingestion import ingest_path
from rag_assistant.retrieval import (
    CrossEncoderReranker,
    LexicalReranker,
    RerankerModelError,
    Retriever,
    best,
    get_reranker,
    make_retriever,
)

SAMPLE = Path(__file__).parent.parent / "data" / "sample"


def src(text: str, score: float = 0.5) -> Source:
    return Source(text, f"{text[:8]}.md", text[:8], score)


def test_best_keeps_the_highest_scores_and_records_them():
    sources = [src("a"), src("b"), src("c")]
    top = best(sources, [0.1, 0.9, 0.5], top_k=2)
    assert [s.text for s in top] == ["b", "c"]
    assert [s.rerank_score for s in top] == [0.9, 0.5]
    assert sources[1].rerank_score is None  # the originals are left alone


def test_best_breaks_ties_by_the_original_order():
    assert [s.text for s in best([src("a"), src("b")], [1.0, 1.0], top_k=2)] == ["a", "b"]


def test_best_checks_its_inputs():
    with pytest.raises(ValueError, match="top_k"):
        best([src("a")], [1.0], top_k=0)
    with pytest.raises(ValueError, match="one score per source"):
        best([src("a")], [], top_k=1)


def test_lexical_reranker_promotes_exact_keyword_matches():
    # Vector search put a generic chunk first; the exact term is in the second.
    sources = [
        src("Indexes make search fast."),
        src("Qdrant builds an HNSW index."),
        src("An index speeds up lookups."),
    ]
    top = LexicalReranker().rerank("HNSW index", sources, top_k=3)
    assert [s.text for s in top] == [
        "Qdrant builds an HNSW index.",  # 2nd by vectors, 1st by keywords
        "Indexes make search fast.",  # 1st by vectors, 3rd by keywords
        "An index speeds up lookups.",  # 3rd by vectors, 2nd by keywords
    ]


def test_lexical_reranker_ties_go_to_vector_order():
    # Two chunks ranked in opposite orders score the same under RRF.
    sources = [src("Indexes make search fast."), src("Qdrant builds an HNSW index.")]
    top = LexicalReranker().rerank("HNSW index", sources, top_k=2)
    assert top[0].rerank_score == top[1].rerank_score
    assert top[0].text == "Indexes make search fast."


def test_lexical_reranker_keeps_vector_order_without_keywords():
    sources = [src("first"), src("second")]
    assert [s.text for s in LexicalReranker().rerank("zzz", sources, top_k=2)] == [
        "first",
        "second",
    ]


def fake_fastembed(monkeypatch, cross_encoder: object) -> None:
    """Stand-ins for the fastembed modules, so no model is downloaded."""
    module = ModuleType("fastembed.rerank.cross_encoder")
    module.TextCrossEncoder = cross_encoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", ModuleType("fastembed"))
    monkeypatch.setitem(sys.modules, "fastembed.rerank", ModuleType("fastembed.rerank"))
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", module)


class LengthScorer:
    """A pretend cross-encoder: longer passages score higher."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def rerank(self, query, documents):
        return [float(len(d)) for d in documents]


def test_cross_encoder_scores_every_pair(monkeypatch):
    fake_fastembed(monkeypatch, LengthScorer)
    reranker = CrossEncoderReranker("some/model")
    top = reranker.rerank("q", [src("short"), src("much longer text")], top_k=1)
    assert top[0].text == "much longer text"
    assert top[0].rerank_score == 16.0
    assert reranker.rerank("q", [], top_k=1) == []


def test_cross_encoder_explains_a_failed_download(monkeypatch):
    def offline(model_name):
        raise ValueError(f"Could not load model {model_name} from any source.")

    fake_fastembed(monkeypatch, offline)
    with pytest.raises(RerankerModelError, match="RAG_RERANKER=lexical"):
        CrossEncoderReranker("some/model")


def test_get_reranker_follows_the_settings(monkeypatch):
    fake_fastembed(monkeypatch, LengthScorer)
    assert isinstance(get_reranker(Settings(embedder="hashing")), LexicalReranker)
    assert isinstance(get_reranker(Settings(embedder="fastembed")), CrossEncoderReranker)
    assert isinstance(get_reranker(Settings(reranker="lexical")), LexicalReranker)
    assert get_reranker(Settings(reranker="none")) is None


def test_retriever_fetches_candidates_then_keeps_top_k(store):
    ingest_path(SAMPLE, store)
    seen: list[int] = []

    class CountingReranker:
        def rerank(self, query, sources, top_k):
            seen.append(len(sources))
            return sources[:top_k]

    found = Retriever(store, top_k=2, reranker=CountingReranker(), candidates=6)("vectors")
    assert len(found) == 2
    assert seen == [6]  # the reranker chose from six candidates


def test_retriever_without_reranker_returns_vector_order(store):
    ingest_path(SAMPLE, store)
    found = Retriever(store, top_k=3)("cosine similarity")
    assert len(found) == 3
    assert [s.score for s in found] == sorted((s.score for s in found), reverse=True)
    assert all(s.rerank_score is None for s in found)


def test_retriever_on_an_empty_store_skips_the_reranker(store):
    class Unused:
        def rerank(self, query, sources, top_k):
            raise AssertionError("nothing to rerank")

    assert Retriever(store, reranker=Unused())("anything") == []


def test_make_retriever_uses_the_settings(store):
    settings = Settings(embedder="hashing", top_k=3, rerank_candidates=9, min_score=0.1)
    retriever = make_retriever(settings, store)
    assert (retriever.top_k, retriever.candidates, retriever.min_score) == (3, 9, 0.1)
    assert isinstance(retriever.reranker, LexicalReranker)
    assert make_retriever(settings, store, top_k=7, rerank=False).reranker is None
    assert make_retriever(settings, store, top_k=7).top_k == 7


def test_lexical_reranking_end_to_end(store):
    ingest_path(SAMPLE, store)
    retriever = Retriever(store, top_k=1, reranker=LexicalReranker(), candidates=10)
    [top] = retriever("Which index does Qdrant use?")
    assert top.source.endswith("vector-databases.md")
    assert top.rerank_score is not None
