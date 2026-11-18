# Agentic RAG Knowledge Assistant

An assistant that answers questions from your own documents, decides where to
search (documents, web or APIs), checks that its answer is supported, and cites
its sources.

> 📚 **Learning this?** [`LEARNING.md`](LEARNING.md) explains every step: what
> was built, why, and the concepts behind it, with exercises.

## Status

- [x] **Step 1: Ingestion.** Load PDF, Markdown, text and HTML; recursive chunking with overlap; local embeddings; Qdrant storage; idempotent re-ingest
- [ ] Step 2: Answers with citations
- [ ] Step 3: Agentic layer (routing, ranking, self-check)
- [ ] Step 4: Chat UI and Docker
- [ ] Step 5: Evaluation

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rag ingest data/sample                 # first run downloads the embedding model (~130 MB)
rag search "what is cosine similarity?"
rag stats
pytest                                 # runs offline in under a second
```

Settings are environment variables; see [`.env.example`](.env.example).

## Architecture (so far)

```
files ─► loaders ─► recursive chunker ─► embedder (bge-small) ─► Qdrant
                                                                    ▲
                                       rag search "question" ───────┘
```

## Tech stack

Python · Qdrant · fastembed (BAAI/bge-small-en-v1.5) · pypdf · BeautifulSoup · Typer · pytest · GitHub Actions

## License

MIT
