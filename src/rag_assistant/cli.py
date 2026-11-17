"""Command line interface.

rag ingest data/sample      # load, chunk, embed and store documents
rag search "what is RAG?"   # semantic search, prints the best chunks
rag ask "what is RAG?"      # answer with citations
rag stats                   # how many chunks are stored
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from contextlib import contextmanager

import typer

from rag_assistant.answering import Answer, answer_question, get_answerer
from rag_assistant.config import get_settings
from rag_assistant.embeddings import get_embedder
from rag_assistant.ingestion import ingest_path
from rag_assistant.vectorstore import VectorStore, make_client

app = typer.Typer(help="Agentic RAG knowledge assistant", no_args_is_help=True)


@contextmanager
def _open_store() -> Iterator[VectorStore]:
    settings = get_settings()
    client = make_client(settings)
    try:
        yield VectorStore(client, settings.collection, get_embedder(settings))
    finally:
        client.close()  # embedded Qdrant locks its folder until the client closes


@app.command()
def ingest(path: str = typer.Argument(..., help="File or folder to ingest")) -> None:
    """Load, chunk, embed and store documents."""
    settings = get_settings()
    with _open_store() as store:
        report = ingest_path(path, store, settings.chunk_size, settings.chunk_overlap)
    typer.echo(
        f"Ingested {report.files} file(s), {report.documents} document(s), "
        f"{report.chunks} chunk(s)."
    )


@app.command()
def search(query: str, top_k: int = typer.Option(5, "--top-k", "-k")) -> None:
    """Show the chunks most similar to QUERY."""
    with _open_store() as store:
        hits = store.search(query, top_k)
    for i, hit in enumerate(hits, start=1):
        where = hit.source + (f", page {hit.metadata['page']}" if "page" in hit.metadata else "")
        preview = hit.text[:200].replace("\n", " ")
        typer.echo(f"[{i}] {hit.score:.3f}  {where}\n    {preview}...\n")


@app.command()
def ask(
    question: str,
    top_k: int = typer.Option(None, "--top-k", "-k", help="Chunks to retrieve (default RAG_TOP_K)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show model, tokens and sources"),
) -> None:
    """Answer QUESTION from your documents, with citations."""
    import anthropic

    from rag_assistant.answering.claude import MissingCredentialsError

    settings = get_settings()
    try:
        answerer = get_answerer(settings)
        with _open_store() as store:
            answer = answer_question(
                question, store, answerer, top_k or settings.top_k, settings.min_score
            )
    except MissingCredentialsError as e:
        _fail(str(e))
    except anthropic.AuthenticationError:
        _fail("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.")
    except anthropic.PermissionDeniedError:
        _fail(f"This API key can't use the model '{settings.llm_model}' (permission denied).")
    except anthropic.NotFoundError:
        _fail(f"Model '{settings.llm_model}' was not found. Check RAG_LLM_MODEL.")
    except anthropic.RateLimitError:
        _fail("Rate limited by the Anthropic API. Wait a moment and try again.")
    except anthropic.APIConnectionError:
        _fail("Couldn't reach the Anthropic API. Check your internet connection.")
    except anthropic.APIStatusError as e:
        _fail(f"Anthropic API error {e.status_code}: {e.message}")

    _print_answer(answer, verbose)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _print_answer(answer: Answer, verbose: bool) -> None:
    typer.echo(answer.render())
    if answer.truncated:
        typer.secho("\n(The answer was cut off: raise RAG_LLM_MAX_TOKENS.)", fg="yellow")
    if answer.fallback_from:
        typer.secho(
            f"\n({answer.fallback_from} declined; answered by {answer.model}.)", fg="yellow"
        )

    cited = answer.cited_source_indices()
    if cited:
        typer.echo("\nSources:")
        for n, idx in enumerate(cited, start=1):
            src = answer.sources[idx]
            typer.echo(f"[{n}] {src.label}  ({src.title}, similarity {src.score:.2f})")
            if verbose:  # the exact passages the answer relied on
                quotes = (
                    c.cited_text
                    for seg in answer.segments
                    for c in seg.citations
                    if c.source_index == idx
                )
                for q in dict.fromkeys(quotes):  # unique, keeps order
                    typer.echo(textwrap.indent(textwrap.fill(f'"{q}"', 88), "    "))

    if verbose:
        uncited = [s for i, s in enumerate(answer.sources) if i not in cited]
        if uncited:
            typer.echo("\nRetrieved but not cited:")
            for src in uncited:
                typer.echo(f"  - {src.label}  (similarity {src.score:.2f})")
        typer.echo(
            f"\nModel: {answer.model or 'none'} | tokens in/out: "
            f"{answer.input_tokens}/{answer.output_tokens} | "
            f"cited: {answer.coverage:.0%} of the answer"
        )


@app.command()
def stats() -> None:
    """Print how many chunks are stored."""
    settings = get_settings()
    with _open_store() as store:
        n = store.count()
    typer.echo(f"{n} chunk(s) in collection '{settings.collection}'.")


if __name__ == "__main__":
    app()
