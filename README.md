# Agentic RAG Knowledge Assistant

An assistant that answers questions from your own documents, decides where to
search (documents, web or APIs), checks that its answer is supported, and cites
its sources.

> 📚 **Learning this?** [`LEARNING.md`](LEARNING.md) explains every step: what
> was built, why, and the concepts behind it, with exercises.

## Status

- [x] **Step 1: Ingestion.** Load PDF, Markdown, text and HTML; recursive chunking with overlap; local embeddings; Qdrant storage; idempotent re-ingest
- [x] **Step 2: Answers with citations.** Claude with native `search_result` citations (sentence-level quotes), refusal and fallback handling, an offline extractive baseline, `rag ask`
- [ ] Step 3: Agentic layer (routing, ranking, self-check)
- [ ] Step 4: Chat UI and Docker
- [ ] Step 5: Evaluation

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rag ingest data/sample                 # first run downloads the embedding model (~130 MB)
rag search "what is cosine similarity?"

export ANTHROPIC_API_KEY=sk-ant-...    # or put it in .env
rag ask "How does Qdrant find similar vectors quickly?" -v
RAG_LLM=extractive rag ask "..."       # offline, no API key needed

rag stats
pytest                                 # runs offline in about a second
```

Settings are environment variables; see [`.env.example`](.env.example).

## Architecture (so far)

```
INDEXING   files ─► loaders ─► recursive chunker ─► embedder (bge-small) ─► Qdrant
                                                                              │
QUERY      rag ask "question" ─► embed ─► top-k search ◄──────────────────────┘
                                               │
                                               ▼
                     Claude (search_result blocks, citations on) ─► answer + [n] citations
```

## Tech stack

Python · Claude API (Claude Opus 5.5, native citations) · Qdrant · fastembed (BAAI/bge-small-en-v1.5) · pypdf · BeautifulSoup · Typer · pytest · GitHub Actions

## License

MIT
