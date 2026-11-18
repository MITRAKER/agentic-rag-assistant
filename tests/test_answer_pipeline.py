from pathlib import Path

from rag_assistant.answering import NO_RESULTS, answer_question, get_answerer
from rag_assistant.answering.extractive import ExtractiveAnswerer
from rag_assistant.config import Settings
from rag_assistant.ingestion import ingest_path

SAMPLE = Path(__file__).parent.parent / "data" / "sample"


class SpyAnswerer:
    def __init__(self):
        self.calls = 0

    def answer(self, question, sources):
        self.calls += 1
        raise AssertionError("should not be called")


def test_end_to_end_with_extractive_answerer(store):
    ingest_path(SAMPLE, store)
    answer = answer_question("What does Qdrant use HNSW for?", store, ExtractiveAnswerer())
    assert "HNSW" in answer.text
    cited = [answer.sources[i] for i in answer.cited_source_indices()]
    assert cited[0].source.endswith("vector-databases.md")


def test_empty_store_returns_no_results_without_calling_the_model(store):
    spy = SpyAnswerer()
    answer = answer_question("anything", store, spy)
    assert answer.text == NO_RESULTS
    assert spy.calls == 0


def test_min_score_filters_weak_matches(store):
    ingest_path(SAMPLE, store)
    spy = SpyAnswerer()
    answer = answer_question("zzz qqq", store, spy, min_score=0.99)
    assert answer.text == NO_RESULTS


def test_get_answerer_extractive():
    settings = Settings(llm="extractive")
    assert isinstance(get_answerer(settings), ExtractiveAnswerer)


def test_get_answerer_claude_uses_key_from_settings():
    from rag_assistant.answering.claude import ClaudeAnswerer

    settings = Settings(llm="claude", anthropic_api_key="sk-test", llm_effort="low")
    answerer = get_answerer(settings)
    assert isinstance(answerer, ClaudeAnswerer)
    assert answerer.effort == "low"
    assert answerer.client.api_key == "sk-test"


def test_missing_credentials_gives_a_clear_error(monkeypatch, tmp_path):
    import pytest

    from rag_assistant.answering.claude import MissingCredentialsError, make_client

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile either
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with pytest.raises(MissingCredentialsError, match="RAG_LLM=extractive"):
        make_client()
