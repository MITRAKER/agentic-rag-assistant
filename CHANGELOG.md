# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Step 3 of [LEARNING.md](LEARNING.md): the agentic layer.

### Added

- An agent (`rag ask --agent`). Claude chooses where to look for each
  question, in a loop with hard limits: the user's documents
  (`search_documents`), live package data from the PyPI JSON API
  (`lookup_python_package`), and web search and web fetch, which run on
  Anthropic's servers. Every tool result is citable, so each quote in the
  answer comes from something a tool really returned. `rag ask --agent -v`
  prints the steps the agent took.
- A self-check. A second, smaller model reads each statement of the answer
  next to the passages it cites and returns a verdict in a fixed JSON shape.
  When the answer isn't supported, the agent revises it with that feedback.
- Reranking. Vector search fetches more candidates than it needs
  (`RAG_RERANK_CANDIDATES`, 20 by default) and a reranker keeps the best
  `RAG_TOP_K`: a local cross-encoder (ms-marco-MiniLM-L-6-v2 through
  fastembed, an 80 MB download on first use) or, offline, a lexical reranker
  that fuses the vector order with keyword overlap. `rag search` prints both
  scores, and `--no-rerank` shows the vector order alone.
- Settings for all of this, each validated at start-up: `RAG_RERANKER`,
  `RAG_RERANK_MODEL`, `RAG_RERANK_CANDIDATES`, `RAG_AGENT_MAX_TURNS`,
  `RAG_AGENT_TOOL_BUDGET`, `RAG_AGENT_WEB`, `RAG_AGENT_PYPI`,
  `RAG_AGENT_SELF_CHECK`, `RAG_AGENT_CHECK_MODEL` and `RAG_AGENT_RETRIES`.
  See [.env.example](.env.example).
- Safety limits for the agent: a per-question cap on replies and on uses of
  each tool, a fixed host, name check, timeout and 5 MB reply cap for PyPI,
  and web fetch restricted to URLs the user typed or web search found.

### Changed

- `rag ask` and `rag search` rerank by default (`RAG_RERANKER=auto`). Set
  `RAG_RERANKER=none` for the previous behaviour.
- `answer_question()` takes a retriever (any function from a query to a list
  of sources) instead of a vector store, `top_k` and `min_score`, and
  `rag_assistant.answering.retrieve()` is replaced by
  `rag_assistant.retrieval.Retriever`.

## [0.2.0] - 2026-11-17

The first release. It contains steps 1 and 2 of [LEARNING.md](LEARNING.md) and
the engineering standards every later step follows. It needs Python 3.11 or
newer (Python 3.10 reached end of life in October 2026).

### Added

- Ingestion of PDF, Markdown, text and HTML files: recursive chunking with
  sentence-aligned overlap, local embeddings (BAAI/bge-small-en-v1.5 through
  fastembed), and idempotent storage in Qdrant (`rag ingest`, `rag search`,
  `rag stats`).
- Answers with citations from Claude (`rag ask`), using native `search_result`
  blocks so every quote is copied from your documents. Refusals, truncated
  answers and model fallbacks (sticky-routed ones included) are reported.
  Requests are streamed, so `RAG_LLM_MAX_TOKENS` can go up to 128,000. An
  offline extractive answerer works without an API key (`RAG_LLM=extractive`).
- Settings are validated when the program starts (for example
  `RAG_TOP_K=0` is rejected with a clear message). API errors and timeouts,
  broken credential settings and a failed embedding-model download are each
  explained in one line instead of a traceback.
- The offline hashing embedder and the extractive answerer match words in any
  script, not only English letters.
- Ingestion limits: at most 10,000 files per run and 100 MB per file.
- CI on every pull request: ruff, mypy (strict), the NASA/JPL Power of Ten
  checks, tests on Python 3.11 to 3.14 with at least 90% branch coverage,
  Bandit, pip-audit, a package build and an offline smoke test of the built
  wheel.
- CodeQL code scanning, Dependabot updates, pre-commit hooks, and a release
  workflow that publishes a GitHub release when a version tag is pushed.
- CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, issue forms, a pull request
  template and CODEOWNERS.

[Unreleased]: https://github.com/MITRAKER/agentic-rag-assistant/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/MITRAKER/agentic-rag-assistant/releases/tag/v0.2.0
