"""Command line interface.

rag ingest data/sample      # load, chunk, embed and store documents
rag search "what is RAG?"   # semantic search, prints the best chunks
rag stats                   # how many chunks are stored
"""

from __future__ import annotations

import typer

from rag_assistant.config import get_settings
from rag_assistant.embeddings import get_embedder
from rag_assistant.ingestion import ingest_path
from rag_assistant.vectorstore import VectorStore, make_client

app = typer.Typer(help="Agentic RAG knowledge assistant", no_args_is_help=True)


def _store() -> VectorStore:
    settings = get_settings()
    return VectorStore(make_client(settings), settings.collection, get_embedder(settings))


@app.command()
def ingest(path: str = typer.Argument(..., help="File or folder to ingest")) -> None:
    """Load, chunk, embed and store documents."""
    settings = get_settings()
    report = ingest_path(path, _store(), settings.chunk_size, settings.chunk_overlap)
    typer.echo(
        f"Ingested {report.files} file(s), {report.documents} document(s), "
        f"{report.chunks} chunk(s)."
    )


@app.command()
def search(query: str, top_k: int = typer.Option(5, "--top-k", "-k")) -> None:
    """Show the chunks most similar to QUERY."""
    for i, hit in enumerate(_store().search(query, top_k), start=1):
        where = hit.source + (f", page {hit.metadata['page']}" if "page" in hit.metadata else "")
        preview = hit.text[:200].replace("\n", " ")
        typer.echo(f"[{i}] {hit.score:.3f}  {where}\n    {preview}...\n")


@app.command()
def stats() -> None:
    """Print how many chunks are stored."""
    typer.echo(f"{_store().count()} chunk(s) in collection '{get_settings().collection}'.")


if __name__ == "__main__":
    app()
