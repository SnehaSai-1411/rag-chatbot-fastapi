"""Shared fixtures: fake embedder, ephemeral Chroma collections.

Everything here runs offline with no model download and no API key.
"""
import hashlib
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = REPO_ROOT / "data" / "sample_docs"

DIM = 128


def fake_embed(text: str) -> list[float]:
    """Deterministic hash-bag embedding: tokens hash into a fixed vector.

    Texts sharing words get similar vectors, which is all the tests need
    to exercise retrieval ranking without a real model.
    """
    vec = [0.0] * DIM
    for token in text.lower().split():
        token = "".join(ch for ch in token if ch.isalnum())
        if not token:
            continue
        idx = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture()
def collection(tmp_path):
    """A real Chroma collection with the sample docs indexed (fake embedder)."""
    import chromadb

    from app.ingest import ingest_folder

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    coll = client.get_or_create_collection(
        name="test-docs", metadata={"hnsw:space": "cosine"}
    )
    report = ingest_folder(SAMPLE_DOCS, coll, fake_embed)
    assert report["chunks_indexed"] > 0
    return coll


@pytest.fixture()
def retriever(collection):
    from app.retrieval import HybridRetriever

    return HybridRetriever(collection, fake_embed)


@pytest.fixture()
def history_store(tmp_path):
    from app.history import HistoryStore

    return HistoryStore(str(tmp_path / "history.db"), max_turns=10)


@pytest.fixture()
def api_client(tmp_path, retriever, history_store, monkeypatch):
    """TestClient with the real retriever, a tmp SQLite history store,
    and a fake LLM (no API key, no model)."""
    import app.main as main_module
    from app.retrieval import citations_for
    from fastapi.testclient import TestClient

    def fake_generate_answer(question, chunks, settings, history=None, llm_fn=None):
        return {
            "answer": f"Fake answer to: {question}",
            "citations": citations_for(chunks),
            "provider": "fake",
            "fallback": False,
        }

    monkeypatch.setattr(main_module, "get_retriever", lambda: retriever)
    monkeypatch.setattr(main_module, "get_history_store", lambda: history_store)
    monkeypatch.setattr(main_module, "generate_answer", fake_generate_answer)
    return TestClient(main_module.app)
