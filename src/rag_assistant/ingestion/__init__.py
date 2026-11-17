"""Turning raw files into searchable chunks: load -> chunk -> embed -> store."""

from rag_assistant.ingestion.chunking import Chunk, chunk_document
from rag_assistant.ingestion.loaders import Document, IngestionError, load_file, load_path
from rag_assistant.ingestion.pipeline import IngestReport, ingest_path

__all__ = (
    "Chunk",
    "Document",
    "IngestReport",
    "IngestionError",
    "chunk_document",
    "ingest_path",
    "load_file",
    "load_path",
)
