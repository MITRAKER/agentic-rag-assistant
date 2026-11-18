"""Central configuration, read from environment variables (or a .env file).

Every setting is prefixed with RAG_, e.g. RAG_CHUNK_SIZE=1000.
Keeping config in one typed object means no magic strings scattered around
the code, and tests can build a Settings(...) with whatever values they need.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Vector database
    qdrant_url: str | None = None  # e.g. http://localhost:6333; None -> embedded mode
    qdrant_path: str = "./qdrant_data"  # used only in embedded mode
    collection: str = "documents"

    # Embeddings
    embedder: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking (characters)
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5  # how many chunks to hand to the answerer
    min_score: float = 0.0  # drop chunks less similar than this (0 = keep all)

    # Answering
    llm: Literal["claude", "extractive"] = "claude"
    llm_model: str = "claude-opus-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    llm_max_tokens: int = 16000
    llm_fallbacks: bool = True  # let the API retry a declined request on a fallback model

    # Read from ANTHROPIC_API_KEY (environment or .env). SecretStr keeps the key
    # out of logs and tracebacks: printing it shows '**********'. When empty, the
    # Anthropic SDK falls back to its own lookup (env vars, `ant auth login`).
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")


def get_settings() -> Settings:
    return Settings()
