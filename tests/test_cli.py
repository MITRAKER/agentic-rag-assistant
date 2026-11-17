from pathlib import Path

import anthropic
import pytest
from typer.testing import CliRunner

from rag_assistant.answering import Answer, Segment
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
    assert result.exit_code == 0
    assert "[1]" in result.output

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


def test_invalid_setting_is_named_without_a_traceback(offline_env, monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "0")
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 1
    assert "Invalid settings: RAG_TOP_K: Input should be greater than or equal to 1" in (
        result.output
    )
    assert "Traceback" not in result.output


def test_settings_that_contradict_each_other_are_reported(offline_env, monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")
    result = runner.invoke(app, ["ingest", SAMPLE])
    assert result.exit_code == 1
    assert "Invalid settings: RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE" in (
        result.output
    )


class FakeResponse:
    """Just enough of an HTTP response for the SDK's error classes."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.request = object()
        self.headers: dict[str, str] = {}


def status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    return cls("boom", response=FakeResponse(status), body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (status_error(anthropic.AuthenticationError, 401), "rejected the API key"),
        (status_error(anthropic.PermissionDeniedError, 403), "permission denied"),
        (status_error(anthropic.NotFoundError, 404), "Check RAG_LLM_MODEL"),
        (status_error(anthropic.RateLimitError, 429), "Rate limited"),
        (anthropic.APIConnectionError(request=object()), "Couldn't reach"),
        (status_error(anthropic.InternalServerError, 500), "Anthropic API error 500: boom"),
    ],
)
def test_ask_explains_api_errors(offline_env, monkeypatch, error, message):
    class FailingAnswerer:
        def answer(self, question, sources):
            raise error

    monkeypatch.setattr("rag_assistant.cli.get_answerer", lambda settings: FailingAnswerer())
    assert runner.invoke(app, ["ingest", SAMPLE]).exit_code == 0
    result = runner.invoke(app, ["ask", "What does Qdrant use HNSW for?"])
    assert result.exit_code == 1
    assert message in result.output


def test_ask_warns_about_cut_off_and_fallback_answers(offline_env, monkeypatch):
    def cut_off(question, store, answerer, top_k, min_score):
        return Answer(
            question,
            [Segment("Half an ans")],
            [],
            model="fallback-model",
            stop_reason="max_tokens",
            fallback_from="claude-opus-5-5",
        )

    monkeypatch.setattr("rag_assistant.cli.answer_question", cut_off)
    result = runner.invoke(app, ["ask", "anything"])
    assert result.exit_code == 0
    assert "cut off: raise RAG_LLM_MAX_TOKENS" in result.output
    assert "claude-opus-5-5 declined; answered by fallback-model" in result.output


def test_ingest_reports_a_missing_path(offline_env):
    result = runner.invoke(app, ["ingest", "no/such/folder"])
    assert result.exit_code == 1
    assert "No such file or folder: no/such/folder" in result.output


def test_ingest_reports_an_unsupported_file(offline_env, tmp_path):
    notes = tmp_path / "notes.docx"
    notes.write_bytes(b"not really a docx")
    result = runner.invoke(app, ["ingest", str(notes)])
    assert result.exit_code == 1
    assert "Unsupported file type: notes.docx" in result.output
