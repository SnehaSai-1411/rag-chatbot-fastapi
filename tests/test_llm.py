"""LLM tests: extractive fallback when no key is set, injected LLM,
and prompt construction."""
from app.config import Settings
from app.llm import FALLBACK_LABEL, build_prompt, generate_answer
from app.retrieval import RetrievedChunk, citations_for


def _chunk(source="chunking-strategies.md", page=1, text="Chunking splits documents."):
    return RetrievedChunk(
        id="x", text=text, source=source, page=page, chunk_index=0, score=0.1
    )


def _no_key_settings():
    return Settings(llm_provider="claude", anthropic_api_key="", openai_api_key="")


def test_fallback_when_no_key():
    result = generate_answer(
        "What is chunking?", [_chunk()], _no_key_settings()
    )
    assert result["fallback"] is True
    assert result["provider"] == "extractive-fallback"
    assert FALLBACK_LABEL.split("—")[0].strip() in result["answer"]
    assert "Chunking splits documents." in result["answer"]
    assert result["citations"] == [{"source": "chunking-strategies.md", "page": 1}]


def test_fallback_with_no_chunks_is_honest():
    result = generate_answer("Anything?", [], _no_key_settings())
    assert result["fallback"] is True
    assert "No relevant context was retrieved" in result["answer"]
    assert result["citations"] == []


def test_injected_llm_is_used_instead_of_fallback():
    result = generate_answer(
        "What is chunking?",
        [_chunk()],
        _no_key_settings(),
        llm_fn=lambda prompt: "INJECTED ANSWER",
    )
    assert result["answer"] == "INJECTED ANSWER"
    assert result["fallback"] is False
    assert result["provider"] == "injected"


def test_build_prompt_includes_context_and_history():
    prompt = build_prompt(
        "Why chunk?",
        [_chunk()],
        history=[{"role": "user", "content": "What is RAG?"}],
    )
    assert "Chunking splits documents." in prompt
    assert "chunking-strategies.md" in prompt
    assert "Why chunk?" in prompt
    assert "What is RAG?" in prompt


def test_citations_deduped_in_order():
    chunks = [
        _chunk(source="a.md", page=1),
        _chunk(source="a.md", page=1, text="more a"),
        _chunk(source="b.md", page=2),
    ]
    assert citations_for(chunks) == [
        {"source": "a.md", "page": 1},
        {"source": "b.md", "page": 2},
    ]
