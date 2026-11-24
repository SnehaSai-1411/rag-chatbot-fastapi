# rag-chatbot-fastapi

A RAG answering service: ask a question, get a grounded answer with citations — FastAPI backend, hybrid retrieval, SQLite conversation history, Streamlit UI.

## What it does
- `POST /ask` — embeds the question, retrieves the top-k chunks, builds a grounded prompt, and answers with Claude or OpenAI, citing source + page for every claim
- Hybrid retrieval — Chroma vector search plus BM25 keyword search over the same chunks, merged with reciprocal rank fusion
- Conversation history — the last N turns per session (default 10) in SQLite, passed back as context
- Honest fallback — with no LLM API key configured, `/ask` returns a clearly labeled extractive excerpt from the retrieved chunks instead of pretending to be a generated answer
- Reuses Project 1's collection schema (chunk ids, metadata, cosine space), so an index built by `rag-data-ingestion-pipeline` works here unchanged

## Architecture

```
User ──▶ Streamlit UI ──▶ FastAPI /ask ──▶ HybridRetriever ──▶ Chroma (vector)
                │                              │   └─▶ BM25 (keyword)
                │                              ▼
                │                       RRF merge ──▶ grounded prompt
                │                              ▼
                │                       Claude / OpenAI
                │                       (or extractive fallback)
                ▼
        answer + citations        SQLite history (last N turns)
```

## Quickstart

```bash
docker compose up
# API at http://localhost:8000, UI at http://localhost:8501
```

Or without Docker:

```bash
pip install -r requirements.txt
python scripts/ingest_sample.py        # builds the sample knowledge base
uvicorn app.main:app                   # API at http://localhost:8000
```

Copy `.env.example` to `.env` and add an `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` (with `LLM_PROVIDER` set to match) to get generated answers.
Without a key, the service still runs and answers with the extractive
fallback. The Streamlit UI runs separately with `streamlit run ui/app.py`.

## Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "session_id": "demo"}'
```

```json
{
  "answer": "Chunking is the step that turns long documents into the small pieces a retrieval system actually searches over...",
  "citations": [
    {"source": "chunking-strategies.md", "page": 1}
  ],
  "session_id": "demo",
  "fallback": false,
  "provider": "claude"
}
```

## Design decisions & trade-offs
- Hybrid retrieval over vector-only: embeddings miss exact terms (codes, rare words) and BM25 misses paraphrases; RRF merging lets a chunk reach the top-k on either signal.
- Local embeddings and Chroma: zero API cost and fully offline-capable; the trade-off is a smaller model and a single-node store, both fine for a personal project.
- SQLite history: simple, file-based, and enough for per-session turns; a multi-user deployment would want Postgres or Redis instead.
- Extractive fallback instead of failing without a key: the service stays demonstrable with no account and no cost, and the label makes it obvious the answer is not LLM-generated.

## What I'd add next
- Reranking of the fused top-k before answering (a cross-encoder pass)
- Streaming answers over SSE so long responses appear token by token
- An evaluation harness with a golden question set (Project 3) to measure retrieval quality instead of eyeballing it
- Guardrails: PII redaction on input and a grounding check on output (Project 4)
