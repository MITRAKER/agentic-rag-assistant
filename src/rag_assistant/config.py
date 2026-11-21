"""Central configuration, read from environment variables (or a .env file).

Every setting is prefixed with RAG_, e.g. RAG_CHUNK_SIZE=1000.
Keeping config in one typed object means no magic strings scattered around
the code, and tests can build a Settings(...) with whatever values they need.
Pydantic validates every value when Settings is created, so a bad value
(say RAG_TOP_K=0) fails immediately with a clear message instead of
causing strange behaviour later. (Power of Ten rule 7: check inputs.)
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Vector database
    qdrant_url: str | None = None  # e.g. http://localhost:6333; None means embedded mode
    qdrant_path: str = "./qdrant_data"  # used only in embedded mode
    collection: str = Field(default="documents", min_length=1)

    # Embeddings
    embedder: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking, measured in characters
    chunk_size: int = Field(default=800, ge=50, le=20_000)
    chunk_overlap: int = Field(default=120, ge=0)

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=100)  # how many chunks to hand to the answerer
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)  # drop weaker chunks; 0 keeps all
    # Reranking: vector search fetches rerank_candidates chunks, the reranker keeps top_k.
    # "auto" uses the cross-encoder with fastembed and the offline lexical reranker with hashing.
    reranker: Literal["auto", "cross-encoder", "lexical", "none"] = "auto"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=200)

    # Answering
    llm: Literal["claude", "extractive"] = "claude"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Effort = "medium"
    llm_max_tokens: int = Field(default=16_000, ge=1, le=128_000)
    llm_fallbacks: bool = True  # let the API retry a declined request on a fallback model

    # Agent (rag ask --agent): Claude chooses its own tools, within these limits
    agent_max_turns: int = Field(default=8, ge=2, le=30)  # model replies per attempt, at most
    agent_tool_budget: int = Field(default=4, ge=1, le=20)  # uses of each tool per attempt
    agent_web: bool = True  # web search and fetch, which run on Anthropic's servers
    agent_pypi: bool = True  # look up Python packages on PyPI: an example of an API tool
    agent_self_check: bool = True  # a second model checks the answer against its sources
    agent_check_model: str = "claude-haiku-5-5"  # small and fast: checking is a narrow task
    agent_retries: int = Field(default=1, ge=0, le=3)  # revisions after a failed check

    # Read from ANTHROPIC_API_KEY (environment or .env). SecretStr keeps the key
    # out of logs and tracebacks: printing it shows '**********'. When empty, the
    # Anthropic SDK falls back to its own lookup (env vars, `ant auth login`).
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    @model_validator(mode="after")
    def _overlap_fits_in_chunk(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        return self


def get_settings() -> Settings:
    return Settings()
