"""Retrieval tests: RRF merge behavior and hybrid retrieval ranking."""
from app.retrieval import reciprocal_rank_fusion


def test_rrf_ranks_dual_top_hit_first():
    # "a" is #1 in both rankings, "b" is #1 in only one.
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["a", "d", "e"]])
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["d"]


def test_rrf_rank_order_within_one_list():
    scores = reciprocal_rank_fusion([["x", "y", "z"]])
    assert scores["x"] > scores["y"] > scores["z"]


def test_rrf_empty_rankings():
    assert reciprocal_rank_fusion([[], []]) == {}


def test_retrieve_finds_chunking_doc(retriever):
    results = retriever.retrieve("What is chunking?", top_k=3)
    assert len(results) == 3
    assert any(r.source == "chunking-strategies.md" for r in results)
    # Scores are fused RRF scores, sorted best first.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_retrieve_finds_embeddings_doc(retriever):
    results = retriever.retrieve("How do embeddings represent meaning?", top_k=3)
    assert any(r.source == "embeddings.md" for r in results)


def test_retrieve_respects_top_k(retriever):
    results = retriever.retrieve("vector search", top_k=2)
    assert len(results) == 2


def test_retrieve_empty_collection_returns_nothing(tmp_path):
    import chromadb

    from app.retrieval import HybridRetriever
    from tests.conftest import fake_embed

    client = chromadb.PersistentClient(path=str(tmp_path / "empty"))
    coll = client.get_or_create_collection(
        name="empty", metadata={"hnsw:space": "cosine"}
    )
    retriever = HybridRetriever(coll, fake_embed)
    assert retriever.chunk_count == 0
    assert retriever.retrieve("anything", top_k=3) == []
