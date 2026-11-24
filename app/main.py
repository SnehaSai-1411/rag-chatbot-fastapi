"""FastAPI backend for the RAG chatbot.

POST /ask — embed the question, retrieve top-k chunks (hybrid vector +
BM25), build a grounded prompt, and answer with an LLM (or an honest
extractive fallback when no API key is configured). Conversation history
is kept per session_id in SQLite.

GET /health — liveness check (never loads the embedding model).
GET /sessions/{session_id}/history — the stored turns for a session.
"""
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .embedder import SentenceTransformerEmbedder
from .history import HistoryStore
from .llm import generate_answer
from .retrieval import HybridRetriever, load_collection

app = FastAPI(title="rag-chatbot-fastapi", version="0.1.0")

# Lazily built singletons — nothing heavy is constructed at import time,
# so /health and the test client stay fast.
_retriever: HybridRetriever | None = None
_history_store: HistoryStore | None = None
_settings: Settings | None = None


def get_settings_cached() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        settings = get_settings_cached()
        collection = load_collection(settings.chroma_dir, settings.collection)
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        _retriever = HybridRetriever(collection, embedder.embed_query)
    return _retriever


def get_history_store() -> HistoryStore:
    global _history_store
    if _history_store is None:
        settings = get_settings_cached()
        _history_store = HistoryStore(settings.history_db, settings.history_turns)
    return _history_store


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="The question to answer.")
    session_id: str | None = Field(
        default=None, description="Conversation session; a new one is created if omitted."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Chunks to retrieve (default from TOP_K)."
    )


class Citation(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str
    fallback: bool
    provider: str


@app.get("/health")
def health() -> dict:
    settings = get_settings_cached()
    return {
        "status": "ok",
        "collection": settings.collection,
        "llm_provider": settings.llm_provider,
        "llm_key_configured": settings.has_llm_key,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    settings = get_settings_cached()
    session_id = request.session_id or str(uuid.uuid4())
    top_k = request.top_k or settings.top_k

    retriever = get_retriever()
    if retriever.chunk_count == 0:
        raise HTTPException(
            status_code=503,
            detail="The knowledge base is empty. Run scripts/ingest_sample.py first.",
        )
    history_store = get_history_store()

    history = history_store.get_history(session_id)
    chunks = retriever.retrieve(request.question, top_k=top_k)
    result = generate_answer(request.question, chunks, settings, history=history)

    history_store.add_turn(session_id, "user", request.question)
    history_store.add_turn(session_id, "assistant", result["answer"])

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        session_id=session_id,
        fallback=result["fallback"],
        provider=result["provider"],
    )


@app.get("/sessions/{session_id}/history")
def session_history(session_id: str) -> dict:
    history_store = get_history_store()
    return {"session_id": session_id, "turns": history_store.get_history(session_id)}


@app.delete("/sessions/{session_id}/history")
def clear_session_history(session_id: str) -> dict:
    history_store = get_history_store()
    history_store.clear(session_id)
    return {"session_id": session_id, "cleared": True}
