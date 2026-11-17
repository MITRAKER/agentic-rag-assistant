import pytest
from pydantic import ValidationError

from rag_assistant.config import Settings


def test_defaults_are_valid():
    settings = Settings()
    assert settings.chunk_overlap < settings.chunk_size


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="RAG_CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize(
    ("field", "value"),
    [("top_k", 0), ("chunk_size", 10), ("min_score", 2.0), ("llm_max_tokens", 0)],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_env_variables_are_read(monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    settings = Settings()
    assert settings.top_k == 7
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-secret"
    assert "sk-secret" not in repr(settings)  # SecretStr keeps it out of logs
