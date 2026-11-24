"""Smoke test: ingest the sample docs into an ephemeral Chroma dir and
retrieve a known chunk top-3. Runs offline with a fake embedder.
"""
import re

from app.ingest import ingest_folder
from app.retrieval import HybridRetriever
from tests.conftest import SAMPLE_DOCS, fake_embed

CHUNK_ID_RE = re.compile(r"^.+?-p\d+-c\d+-[0-9a-f]{12}$")


def test_ingest_and_retrieve_sample_docs(tmp_path):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection(
        name="smoke", metadata={"hnsw:space": "cosine"}
    )
    report = ingest_folder(SAMPLE_DOCS, collection, fake_embed)

    assert report["documents"] == 4
    assert report["chunks_indexed"] == report["chunks_produced"] > 0
    assert report["duplicates_dropped"] == 0

    # Chunk ids and metadata follow the Project 1 schema.
    stored = collection.get(include=["metadatas"])
    assert stored["ids"], "expected indexed chunks"
    for chunk_id, meta in zip(stored["ids"], stored["metadatas"]):
        assert CHUNK_ID_RE.match(chunk_id), f"bad chunk id: {chunk_id}"
        assert {"source", "page", "chunk_index", "content_hash"} <= set(meta)

    retriever = HybridRetriever(collection, fake_embed)
    results = retriever.retrieve("What is chunking and why does it matter?", top_k=3)
    assert len(results) == 3
    assert any(r.source == "chunking-strategies.md" for r in results)
    assert "chunk" in results[0].text.lower() or any(
        "chunk" in r.text.lower() for r in results
    )
