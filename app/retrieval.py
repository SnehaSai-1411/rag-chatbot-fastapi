"""Hybrid retrieval: Chroma vector search + BM25 keyword search, merged
with reciprocal rank fusion (RRF).

The retriever takes an injected embed function, so tests can pass a fake
embedder and never load a real model. chromadb is imported lazily inside
``load_collection`` only — this module never imports it at top level.
"""
import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .embedder import EmbedFn

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int
    score: float = 0.0  # fused RRF score; higher is better


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Merge ranked lists of chunk ids into one fused score per id.

    Each list contributes 1 / (k + rank) per item (rank is 1-based), so an
    item ranked highly by both retrievers outscores one ranked highly by
    only one. ``k`` dampens the effect of rank differences; 60 is the
    standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def load_collection(persist_dir: str, name: str):
    """Open (or create) a Chroma collection. chromadb is imported here,
    lazily, so importing this module never pulls it in."""
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


class HybridRetriever:
    """Vector + BM25 retrieval over one Chroma collection.

    ``collection`` is any object with Chroma's ``query`` / ``get`` methods
    (a real collection, or a fake in tests). ``embed_fn`` maps one query
    string to one embedding vector.
    """

    def __init__(self, collection, embed_fn: EmbedFn):
        self.collection = collection
        self.embed_fn = embed_fn
        self._chunks: dict[str, RetrievedChunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        corpus = []
        for chunk_id, doc, meta in zip(ids, docs, metas):
            meta = meta or {}
            self._chunks[chunk_id] = RetrievedChunk(
                id=chunk_id,
                text=doc,
                source=meta.get("source", ""),
                page=meta.get("page", 0),
                chunk_index=meta.get("chunk_index", 0),
            )
            corpus.append(tokenize(doc))
            self._bm25_ids.append(chunk_id)
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _vector_search(self, query: str, n: int) -> list[str]:
        if not self._chunks:
            return []
        query_embedding = self.embed_fn(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n, len(self._chunks)),
        )
        return list(results.get("ids", [[]])[0])

    def _bm25_search(self, query: str, n: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Only keep chunks with a positive BM25 score — a zero means no
        # query term matched at all.
        return [self._bm25_ids[i] for i in ranked if scores[i] > 0][:n]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k chunks by fused RRF score, best first."""
        vector_ranking = self._vector_search(query, top_k)
        bm25_ranking = self._bm25_search(query, top_k)
        fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
        ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        results = []
        for cid in ranked_ids:
            chunk = self._chunks[cid]
            results.append(
                RetrievedChunk(
                    id=chunk.id, text=chunk.text, source=chunk.source,
                    page=chunk.page, chunk_index=chunk.chunk_index,
                    score=fused[cid],
                )
            )
        return results


def citations_for(chunks: list[RetrievedChunk]) -> list[dict]:
    """Deduplicated [{source, page}] citations, in retrieval order."""
    seen: set[tuple[str, int]] = set()
    citations = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append({"source": chunk.source, "page": chunk.page})
    return citations
