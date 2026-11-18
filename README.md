# Agentic RAG Knowledge Assistant

[![CI](https://github.com/MITRAKER/agentic-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/MITRAKER/agentic-rag-assistant/actions/workflows/ci.yml)
[![CodeQL](https://github.com/MITRAKER/agentic-rag-assistant/actions/workflows/codeql.yml/badge.svg)](https://github.com/MITRAKER/agentic-rag-assistant/actions/workflows/codeql.yml)
[![Python 3.11 to 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![Coverage at least 90%](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen)](pyproject.toml)
[![mypy strict](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](.pre-commit-config.yaml)
[![NASA/JPL Power of Ten](https://img.shields.io/badge/NASA%2FJPL-Power%20of%20Ten-orange)](LEARNING.md#engineering-standards)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An assistant that answers questions from your own documents, decides where to
search (documents, web or APIs), checks that its answer is supported, and cites
its sources.

> 📚 **Learning this?** [`LEARNING.md`](LEARNING.md) explains every step: what
> was built, why, and the concepts behind it, with exercises.

## Status

- [x] **Step 1: Ingestion.** Load PDF, Markdown, text and HTML; recursive chunking with sentence-aligned overlap; local embeddings; Qdrant storage; idempotent re-ingest
- [x] **Step 2: Answers with citations.** Claude with native `search_result` citations (sentence-level quotes), refusal and fallback handling, an offline extractive baseline, `rag ask`
- [x] **Engineering standards.** CI/CD on GitHub Actions, the NASA/JPL Power of Ten rules adapted to Python, strict typing, security scanning, release automation
- [x] **Step 3: Agentic layer.** `rag ask --agent`: Claude chooses its sources (your documents, the web, the PyPI API) in a bounded tool loop; a cross-encoder reranks retrieved chunks; a second model checks the answer against its citations and asks for a revision when it isn't supported
- [ ] Step 4: Chat UI and Docker
- [ ] Step 5: Evaluation

## Quick start

Needs Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rag ingest data/sample                 # first run downloads the embedding model (~130 MB)
rag search "what is cosine similarity?"  # reranked (~80 MB model, once); --no-rerank to compare

export ANTHROPIC_API_KEY=sk-ant-...    # or put it in .env
rag ask "How does Qdrant find similar vectors quickly?" -v
rag ask --agent "Which Qdrant index do my notes describe, and what is the latest qdrant-client?" -v
RAG_LLM=extractive rag ask "..."       # offline, no API key needed

rag stats
pytest                                 # runs offline in a few seconds
```

Settings are environment variables; see [`.env.example`](.env.example). An
invalid value (for example `RAG_TOP_K=0`) stops the program with a message
naming the variable.

## Architecture (so far)

```
INDEXING   files ─► loaders ─► recursive chunker ─► embedder (bge-small) ─► Qdrant
                                                                              │
RETRIEVAL  query ─► embed ─► vector search, 20 candidates ◄───────────────────┘
                 ─► cross-encoder reranker ─► the best top-k chunks

PIPELINE   rag ask "question" ─► retrieval ─► Claude, citations on ─► answer + [n]

AGENT      rag ask --agent "question"
                │
                ▼
             Claude ◄──────────── tool results ─────────────┐
                │                                           │
                ├─ asks for a tool ─► search_documents ─────┤  (the retrieval above)
                │                     lookup_python_package ┤  (the PyPI API)
                │                     web_search, web_fetch ┘  (Anthropic runs these)
                │
                ▼ answers, with citations
             self-check: a second model reads each statement next to its passages
                ├─ supported ─► done
                └─ not supported ─► its feedback goes back to Claude, which revises

           Limits per attempt: RAG_AGENT_MAX_TURNS replies, RAG_AGENT_TOOL_BUDGET uses of each tool
```

The pipeline always retrieves once and answers. The agent decides for itself
where to look, and how often, within hard limits.

## Quality and engineering standards

Every pull request runs the same checks, on GitHub and (through pre-commit)
on your machine before each commit:

| Area | What runs |
|------|-----------|
| Style and bugs | ruff (pycodestyle, pyflakes, bugbear, bandit rules, complexity limits and more) |
| Formatting | ruff format, including the Python examples in Markdown |
| Types | mypy in strict mode |
| Safety rules | [NASA/JPL Power of Ten](LEARNING.md#engineering-standards), adapted to Python and checked by [`tools/power_of_ten.py`](tools/power_of_ten.py) |
| Tests | pytest on Python 3.11, 3.12, 3.13 and 3.14; at least 90% branch coverage; any warning fails the run |
| Security | Bandit and CodeQL for the code, pip-audit and Dependabot for the dependencies |
| Packaging | the wheel is built, installed in a clean environment and used offline |
| Releases | pushing a `vX.Y.Z` tag publishes a GitHub release with notes from [CHANGELOG.md](CHANGELOG.md) |

`make check` runs everything locally. [CONTRIBUTING.md](CONTRIBUTING.md) has
the details.

## Project layout

```
src/rag_assistant/     the package: ingestion/, retrieval.py, answering/, agent/, cli.py, ...
tests/                 pytest suite (offline: in-memory Qdrant, a scripted Claude API)
tools/                 power_of_ten.py (rule checker), release_notes.py
data/sample/           three small Markdown documents to try things on
.github/               CI, CodeQL and release workflows, Dependabot, templates
```

## Tech stack

Python · Claude API (Claude Opus 5.5 with native citations, tool runner, message threads, web search and fetch; Claude Haiku 5.5 with structured outputs for the self-check) · Qdrant · fastembed (BAAI/bge-small-en-v1.5 embeddings, ms-marco-MiniLM cross-encoder) · httpx2 · pypdf · BeautifulSoup · Typer · pydantic-settings · pytest · ruff · mypy · GitHub Actions · CodeQL

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)
