"""Loaders: read files of different formats into plain text + metadata.

Each loader returns a list of `Document`s. A PDF returns one Document per page,
so that later we can cite "report.pdf, page 4" instead of just "report.pdf".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

Metadata: TypeAlias = dict[str, Any]  # JSON-like values: page number, title, ...

SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt", ".html", ".htm"})
# Upper bounds on the work one ingest can do (Power of Ten rules 2 and 3):
# a typo like `rag ingest /` fails fast instead of reading the whole disk.
MAX_FILES = 10_000
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


class IngestionError(ValueError):
    """A file or folder that can't be ingested: unsupported, too big, or too many files."""


@dataclass
class Document:
    """A unit of source text plus where it came from."""

    text: str
    source: str  # file path (or URL later on)
    metadata: Metadata = field(default_factory=dict)  # e.g. {"page": 3, "title": "..."}


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(text=text, source=str(path), metadata={"page": i}))
    return docs


def _load_html(path: Path) -> list[Document]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()  # boilerplate that would only add noise to search
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    text = soup.get_text(separator="\n")
    return [Document(text=text, source=str(path), metadata={"title": title})]


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = path.stem
    # Use the first Markdown heading as the title when there is one.
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return [Document(text=text, source=str(path), metadata={"title": title})]


def load_file(path: str | Path, max_bytes: int = MAX_FILE_BYTES) -> list[Document]:
    path = Path(path)
    size = path.stat().st_size
    if size > max_bytes:
        raise IngestionError(f"{path.name} is {size:,} bytes; the limit is {max_bytes:,}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".html", ".htm"}:
        return _load_html(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return _load_text(path)
    raise IngestionError(f"Unsupported file type: {path.name}")


def find_files(folder: Path, max_files: int = MAX_FILES) -> list[Path]:
    """Every supported file under `folder` (recursively), refusing more than `max_files`."""
    files: list[Path] = []
    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(file)
            if len(files) > max_files:
                raise IngestionError(f"More than {max_files:,} files under {folder}; ingest less")
    return sorted(files)


def load_path(path: str | Path, max_files: int = MAX_FILES) -> list[Document]:
    """Load one file, or every supported file under a directory (recursively)."""
    path = Path(path)
    if path.is_file():
        return load_file(path)
    if not path.exists():
        raise FileNotFoundError(path)
    docs: list[Document] = []
    for file in find_files(path, max_files):
        docs.extend(load_file(file))
    return docs
