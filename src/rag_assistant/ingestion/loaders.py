"""Loaders: read files of different formats into plain text + metadata.

Each loader returns a list of `Document`s. A PDF returns one Document per page,
so that later we can cite "report.pdf, page 4" instead of just "report.pdf".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


@dataclass
class Document:
    """A unit of source text plus where it came from."""

    text: str
    source: str  # file path (or URL later on)
    metadata: dict = field(default_factory=dict)  # e.g. {"page": 3, "title": "..."}


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


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".html", ".htm"}:
        return _load_html(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return _load_text(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_path(path: str | Path) -> list[Document]:
    """Load one file, or every supported file under a directory (recursively)."""
    path = Path(path)
    if path.is_file():
        return load_file(path)
    if not path.exists():
        raise FileNotFoundError(path)
    docs: list[Document] = []
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(load_file(file))
    return docs
