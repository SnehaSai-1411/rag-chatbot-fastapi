"""Index Markdown documents into a Chroma collection.

Uses the Project 1-compatible schema: chunk ids
"{source}-p{page}-c{chunk_index}-{hash12}", metadata source / page /
chunk_index / content_hash, cosine space. Duplicate chunks (same content
hash) are dropped and counted. chromadb is imported lazily inside
``index_chunks`` so this module never pulls it in at import time.
"""
from pathlib import Path

from .chunker import chunk_id, chunk_pages, content_hash, load_markdown_folder


def index_chunks(collection, chunks, embeddings) -> tuple[int, int]:
    """Add (chunk, embedding) pairs. Returns (added, duplicates_dropped)."""
    seen: set[str] = set()
    ids, docs, metas, embs = [], [], [], []
    duplicates = 0
    for chunk, emb in zip(chunks, embeddings):
        h = content_hash(chunk.text)
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        ids.append(chunk_id(chunk.source, chunk.page, chunk.chunk_index, chunk.text))
        docs.append(chunk.text)
        metas.append({
            "source": chunk.source,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "content_hash": h,
        })
        embs.append(emb)
    if ids:
        collection.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
    return len(ids), duplicates


def ingest_folder(source: Path, collection, embed_fn,
                  chunk_size: int = 800, overlap: int = 100) -> dict:
    """Load Markdown files from ``source``, chunk, embed, and index them
    into ``collection``. Returns counts for a small report."""
    pages = load_markdown_folder(source)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    embeddings = [embed_fn(c.text) for c in chunks] if chunks else []
    added, duplicates = index_chunks(collection, chunks, embeddings)
    return {
        "documents": len({p.source for p in pages}),
        "pages": len(pages),
        "chunks_produced": len(chunks),
        "chunks_indexed": added,
        "duplicates_dropped": duplicates,
    }
