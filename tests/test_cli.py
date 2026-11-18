from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_assistant.cli import app

SAMPLE = str(Path(__file__).parent.parent / "data" / "sample")
runner = CliRunner()


@pytest.fixture
def offline_env(monkeypatch, tmp_path):
    """Run the CLI against a throwaway on-disk Qdrant with offline components."""
    monkeypatch.setenv("RAG_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("RAG_EMBEDDER", "hashing")
    monkeypatch.setenv("RAG_LLM", "extractive")
    monkeypatch.chdir(tmp_path)  # so a developer's own .env file is not picked up


def test_ingest_ask_search_stats(offline_env):
    result = runner.invoke(app, ["ingest", SAMPLE])
    assert result.exit_code == 0, result.output
    assert "3 file(s)" in result.output

    result = runner.invoke(app, ["ask", "What does Qdrant use HNSW for?", "-v"])
    assert result.exit_code == 0, result.output
    assert "HNSW" in result.output
    assert "Sources:" in result.output
    assert "vector-databases.md" in result.output
    assert "Model: extractive" in result.output

    result = runner.invoke(app, ["search", "cosine similarity"])
    assert result.exit_code == 0 and "[1]" in result.output

    result = runner.invoke(app, ["stats"])
    assert "chunk(s)" in result.output


def test_ask_without_api_key_explains_what_to_do(offline_env, monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_LLM", "claude")
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = runner.invoke(app, ["ask", "anything"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output
