from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from rich.console import Console

console = Console()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".mdx", ".rst"}
_MIN_CONTENT_LENGTH = 50


@dataclass
class Document:
    content: str
    source: str
    doc_type: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.content = self.content.strip()

    @property
    def is_empty(self) -> bool:
        return len(self.content) < _MIN_CONTENT_LENGTH


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    docs = []
    try:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            doc = Document(
                content=page.extract_text() or "",
                source=str(path),
                doc_type="pdf",
                metadata={"page": i + 1, "total_pages": len(reader.pages)},
            )
            if not doc.is_empty:
                docs.append(doc)
        console.print(f"[green]pdf[/green] {path.name} → {len(docs)} pages")
    except Exception as exc:
        console.print(f"[red]failed to load {path}: {exc}[/red]")
    return docs


def load_text(path: str | Path) -> list[Document]:
    path = Path(path)
    try:
        doc_type = "markdown" if path.suffix in {".md", ".mdx"} else "text"
        doc = Document(
            content=path.read_text(encoding="utf-8", errors="ignore"),
            source=str(path),
            doc_type=doc_type,
        )
        if not doc.is_empty:
            console.print(f"[green]{doc_type}[/green] {path.name} → loaded")
            return [doc]
    except Exception as exc:
        console.print(f"[red]failed to load {path}: {exc}[/red]")
    return []


def load_url(url: str, timeout: int = 10) -> list[Document]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "retrieval-agent/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n"))
        doc = Document(
            content=text,
            source=url,
            doc_type="web",
            metadata={"title": soup.title.string.strip() if soup.title else url},
        )
        if not doc.is_empty:
            console.print(f"[green]web[/green] {url} → {len(text):,} chars")
            return [doc]
    except Exception as exc:
        console.print(f"[red]failed to load {url}: {exc}[/red]")
    return []


def load_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        loader = load_pdf if path.suffix.lower() == ".pdf" else load_text
        docs.extend(loader(path))
    console.print(f"[bold]loaded {len(docs)} documents[/bold]")
    return docs


def load(source: str) -> list[Document]:
    if source.startswith(("http://", "https://")):
        return load_url(source)
    path = Path(source)
    if path.is_dir():
        return load_directory(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return load_text(path)
