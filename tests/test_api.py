"""API tests: /ask returns an answer + citations via TestClient,
with a fake embedder and a fake LLM (no model, no API key)."""


def test_health_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "collection" in body


def test_ask_returns_answer_and_citations(api_client):
    response = api_client.post(
        "/ask",
        json={"question": "What is chunking?", "session_id": "s1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Fake answer to: What is chunking?")
    assert body["session_id"] == "s1"
    assert body["fallback"] is False
    assert body["provider"] == "fake"
    assert body["citations"], "expected at least one citation"
    for cite in body["citations"]:
        assert "source" in cite and "page" in cite
    # The chunking doc should be among the cited sources.
    assert any(c["source"] == "chunking-strategies.md" for c in body["citations"])


def test_ask_creates_session_id_when_omitted(api_client):
    response = api_client.post("/ask", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert response.json()["session_id"]


def test_ask_records_history(api_client):
    api_client.post("/ask", json={"question": "What is chunking?", "session_id": "s2"})
    api_client.post("/ask", json={"question": "And embeddings?", "session_id": "s2"})
    response = api_client.get("/sessions/s2/history")
    assert response.status_code == 200
    turns = response.json()["turns"]
    assert len(turns) == 4  # two user + two assistant messages
    assert turns[0] == {"role": "user", "content": "What is chunking?"}
    assert turns[1]["role"] == "assistant"
    assert turns[2]["content"] == "And embeddings?"


def test_ask_rejects_empty_question(api_client):
    response = api_client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_respects_top_k(api_client):
    response = api_client.post(
        "/ask", json={"question": "What is chunking?", "top_k": 1}
    )
    assert response.status_code == 200
    assert len(response.json()["citations"]) == 1


def test_clear_history(api_client):
    api_client.post("/ask", json={"question": "What is chunking?", "session_id": "s3"})
    response = api_client.delete("/sessions/s3/history")
    assert response.status_code == 200
    assert response.json()["cleared"] is True
    turns = api_client.get("/sessions/s3/history").json()["turns"]
    assert turns == []
