"""Chunking and loading, compatible with Project 1's schema.

Chunk ids are "{source}-p{page}-c{chunk_index}-{hash12}" and metadata carries
source / page / chunk_index / content_hash, so a collection built here can be
read by Project 1's tooling and vice versa. Markdown files are treated as a
single page (page 1) since they have no real pagination.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    source: str  # file name, e.g. "embeddings.md"
    page: int    # 1-based page number (always 1 for Markdown)
    text: str


@dataclass
class Chunk:
    source: str
    page: int
    chunk_index: int  # running index within the source document
    text: str


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    return f"{source}-p{page}-c{chunk_index}-{content_hash(text)[:12]}"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into chunks of at most ~chunk_size characters.

    Paragraphs are kept whole when they fit. A paragraph longer than
    chunk_size is hard-split with a character overlap so context is not
    lost at the cut.
    """
    assert 0 <= overlap < chunk_size, "overlap must be smaller than chunk_size"
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                if end >= len(para):
                    break
                start = end - overlap
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for page in pages:
        if not page.text:
            continue  # empty pages produce no chunks
        for piece in chunk_text(page.text, chunk_size, overlap):
            idx = counters.get(page.source, 0)
            chunks.append(Chunk(source=page.source, page=page.page,
                                chunk_index=idx, text=piece))
            counters[page.source] = idx + 1
    return chunks


def load_markdown(path: Path) -> Page:
    return Page(source=path.name, page=1, text=path.read_text(encoding="utf-8").strip())


def load_markdown_folder(folder: Path) -> list[Page]:
    return [load_markdown(p) for p in sorted(folder.glob("*.md"))]
