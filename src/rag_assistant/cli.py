"""Command line interface.

rag ingest data/sample          # load, chunk, embed and store documents
rag search "what is RAG?"       # search and rerank, prints the best chunks
rag ask "what is RAG?"          # answer with citations
rag ask --agent "what is RAG?"  # let Claude choose its sources and check its answer
rag stats                       # how many chunks are stored
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, NoReturn

import typer
from pydantic import ValidationError

from rag_assistant.answering import Answer, Source, answer_question, get_answerer
from rag_assistant.config import Settings, get_settings
from rag_assistant.embeddings import EmbeddingModelError, get_embedder
from rag_assistant.ingestion import IngestionError, ingest_path
from rag_assistant.retrieval import RerankerModelError, Retriever, make_retriever
from rag_assistant.vectorstore import VectorStore, make_client

if TYPE_CHECKING:
    import anthropic
    from pydantic_core import ErrorDetails

app = typer.Typer(help="Agentic RAG knowledge assistant", no_args_is_help=True)


def _load_settings() -> Settings:
    """Read the settings; on a bad value, name the variable instead of showing a traceback."""
    try:
        return get_settings()
    except ValidationError as e:
        _fail("Invalid settings: " + "; ".join(_describe(err) for err in e.errors()))


def _describe(error: ErrorDetails) -> str:
    """'RAG_TOP_K: Input should be greater than or equal to 1'."""
    field = str(error["loc"][0]) if error["loc"] else ""
    if not field:  # a check across several settings, e.g. overlap vs. chunk size
        return error["msg"].removeprefix("Value error, ")
    variable = field if field.isupper() else f"RAG_{field.upper()}"
    return f"{variable}: {error['msg']}"


@contextmanager
def _open_store(settings: Settings) -> Iterator[VectorStore]:
    try:
        embedder = get_embedder(settings)
    except EmbeddingModelError as e:  # e.g. no internet for the first download
        _fail(str(e))
    client = make_client(settings)
    try:
        yield VectorStore(client, settings.collection, embedder)
    finally:
        client.close()  # embedded Qdrant locks its folder until the client closes


@contextmanager
def _open_retriever(
    settings: Settings, top_k: int | None = None, rerank: bool = True
) -> Iterator[Retriever]:
    with _open_store(settings) as store:
        try:
            retriever = make_retriever(settings, store, top_k, rerank)
        except RerankerModelError as e:  # e.g. no internet for the first download
            _fail(str(e))
        yield retriever


@app.command()
def ingest(path: str = typer.Argument(..., help="File or folder to ingest")) -> None:
    """Load, chunk, embed and store documents."""
    settings = _load_settings()
    try:
        with _open_store(settings) as store:
            report = ingest_path(path, store, settings.chunk_size, settings.chunk_overlap)
    except FileNotFoundError:
        _fail(f"No such file or folder: {path}")
    except IngestionError as e:  # unsupported file type, or over a size or count limit
        _fail(str(e))
    typer.echo(
        f"Ingested {report.files} file(s), {report.documents} document(s), "
        f"{report.chunks} chunk(s)."
    )


@app.command()
def search(
    query: str,
    top_k: int = typer.Option(5, "--top-k", "-k"),
    rerank: bool = typer.Option(
        True, "--rerank/--no-rerank", help="Rerank the vector search results (RAG_RERANKER)"
    ),
) -> None:
    """Show the chunks that best match QUERY, with their similarity and rerank scores."""
    with _open_retriever(_load_settings(), top_k, rerank) as retrieve:
        sources = retrieve(query)
    for i, src in enumerate(sources, start=1):
        where = src.source + (f", page {src.metadata['page']}" if "page" in src.metadata else "")
        reranked = "" if src.rerank_score is None else f"  rerank {src.rerank_score:.3f}"
        preview = src.text[:200].replace("\n", " ")
        typer.echo(f"[{i}] {src.score:.3f}{reranked}  {where}\n    {preview}...\n")


@app.command()
def ask(
    question: str,
    top_k: int = typer.Option(None, "--top-k", "-k", help="Chunks to retrieve (default RAG_TOP_K)"),
    agent: bool = typer.Option(
        False, "--agent", "-a", help="Let Claude choose its sources, then check its answer"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show model, tokens, sources and the agent's steps"
    ),
) -> None:
    """Answer QUESTION from your documents, with citations."""
    import anthropic  # imported here, not at the top: it takes most of a second to load

    from rag_assistant.answering.claude import MissingCredentialsError

    settings = _load_settings()
    if agent and settings.llm != "claude":
        _fail("The agent needs Claude: set RAG_LLM=claude and ANTHROPIC_API_KEY.")
    try:
        answer = _answer(settings, question, top_k, agent)
    except MissingCredentialsError as e:
        _fail(str(e))
    except anthropic.AnthropicError as e:
        _fail(_explain_api_error(e, settings.llm_model))

    _print_answer(answer, verbose)


def _answer(settings: Settings, question: str, top_k: int | None, agent: bool) -> Answer:
    if agent:
        from rag_assistant.agent import get_agent

        with _open_retriever(settings, top_k) as retriever:
            return get_agent(settings, retriever).answer(question)
    answerer = get_answerer(settings)  # first, so missing credentials fail before models load
    with _open_retriever(settings, top_k) as retriever:
        return answer_question(question, retriever, answerer)


def _explain_api_error(error: Exception, model: str) -> str:
    """A short, actionable message for an error raised by the Anthropic SDK."""
    import anthropic

    # Most specific first, because the first matching class wins: for example,
    # APITimeoutError is a kind of APIConnectionError.
    messages: list[tuple[type[Exception], str]] = [
        (anthropic.AuthenticationError, "Anthropic rejected the API key. Check ANTHROPIC_API_KEY."),
        (
            anthropic.PermissionDeniedError,
            "This API key can't use the model '{model}' (permission denied).",
        ),
        (anthropic.NotFoundError, "Model '{model}' was not found. Check RAG_LLM_MODEL."),
        (
            anthropic.RateLimitError,
            "Rate limited by the Anthropic API. Wait a moment and try again.",
        ),
        (
            anthropic.APITimeoutError,
            "The Anthropic API took too long to answer. Try again in a moment.",
        ),
        (
            anthropic.APIConnectionError,
            "Couldn't reach the Anthropic API. Check your internet connection.",
        ),
        (anthropic.CredentialsError, "Couldn't load your Anthropic credentials: {error}"),
    ]
    for kind, message in messages:
        if isinstance(error, kind):
            return message.format(model=model, error=error)
    if isinstance(error, anthropic.APIStatusError):
        return f"Anthropic API error: {_status_error_detail(error)}"
    return f"Anthropic SDK error: {error}"


def _status_error_detail(error: anthropic.APIStatusError) -> str:
    """'overloaded_error: Overloaded', from the error body the API sends.

    An error can also arrive in the middle of a streamed answer, after the
    HTTP status 200 was sent, so the status code alone can mislead.
    """
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("error")
    if isinstance(detail, dict) and "message" in detail:
        return f"{detail.get('type', 'error')}: {detail['message']}"
    return f"{error.message} (HTTP {error.status_code})"


def _fail(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _print_answer(answer: Answer, verbose: bool) -> None:
    typer.echo(answer.render())
    _print_warnings(answer)
    cited = answer.cited_source_indices()
    if cited:
        typer.echo("\nSources:")
        for n, idx in enumerate(cited, start=1):
            typer.echo(f"[{n}] {_describe_source(answer.sources[idx])}")
            if verbose:  # the exact passages the answer relied on
                _print_quotes(answer, idx)
    if not verbose:
        return
    uncited = [s for i, s in enumerate(answer.sources) if i not in cited]
    if uncited:
        typer.echo("\nRetrieved but not cited:")
        for src in uncited:
            typer.echo(f"  - {_describe_source(src)}")
    if answer.steps:
        typer.echo("\nSteps:")
        for n, step in enumerate(answer.steps, start=1):
            typer.echo(f'  {n}. {step.tool} "{step.input}" -> {step.outcome}')
    typer.echo(
        f"\nModel: {answer.model or 'none'} | tokens in/out: "
        f"{answer.input_tokens}/{answer.output_tokens} | "
        f"cited: {answer.coverage:.0%} of the answer"
    )


def _print_warnings(answer: Answer) -> None:
    if answer.truncated:
        typer.secho("\n(The answer was cut off: raise RAG_LLM_MAX_TOKENS.)", fg="yellow")
    if answer.stop_reason in ("tool_use", "pause_turn"):
        typer.secho(
            "\n(The agent used all RAG_AGENT_MAX_TURNS replies before it finished.)", fg="yellow"
        )
    if answer.fallback_from:
        typer.secho(
            f"\n({answer.fallback_from} declined; answered by {answer.model}.)", fg="yellow"
        )
    check = answer.check
    if check and not check.passed:
        typer.secho(f"\n(Self-check: {check.verdict.replace('_', ' ')}.", fg="yellow")
        for claim in check.unsupported_claims:
            typer.secho(f"  Not supported by its sources: {claim}", fg="yellow")
        typer.secho(f"  {check.feedback})" if check.feedback else ")", fg="yellow")


def _describe_source(src: Source) -> str:
    """'vector-databases.md  (Vector Databases, similarity 0.83)' or a URL and its kind."""
    if src.kind == "document":
        return f"{src.label}  ({src.title}, similarity {src.score:.2f})"
    return f"{src.label}  ({src.title}, {'web' if src.kind == 'web' else 'API'})"


def _print_quotes(answer: Answer, idx: int) -> None:
    quotes = (
        c.cited_text for seg in answer.segments for c in seg.citations if c.source_index == idx
    )
    for q in dict.fromkeys(quotes):  # unique, keeps order
        typer.echo(textwrap.indent(textwrap.fill(f'"{q}"', 88), "    "))


@app.command()
def stats() -> None:
    """Print how many chunks are stored."""
    settings = _load_settings()
    with _open_store(settings) as store:
        n = store.count()
    typer.echo(f"{n} chunk(s) in collection '{settings.collection}'.")


if __name__ == "__main__":
    app()
