"""LLM layer: Claude or OpenAI via API key, with an honest extractive
fallback when no key is configured.

The fallback never pretends to be an LLM answer: it returns the top
retrieved chunk excerpt, labeled clearly as an extractive fallback, and
sets ``fallback: True`` so callers (and the UI) can say so too.
"""
from collections.abc import Callable

from .config import Settings
from .retrieval import RetrievedChunk, citations_for

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know — "
    "do not guess or use outside knowledge. "
    "Keep the answer concise and cite the sources you used."
)

FALLBACK_LABEL = (
    "Extractive fallback — no LLM API key is configured, so this is a "
    "verbatim excerpt from the retrieved context, not a generated answer."
)


def build_prompt(question: str, chunks: list[RetrievedChunk],
                 history: list[dict] | None = None) -> str:
    context = "\n\n".join(
        f"[Source: {c.source}, page {c.page}]\n{c.text}" for c in chunks
    )
    parts = []
    if history:
        convo = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        )
        parts.append(f"Previous conversation:\n{convo}")
    parts.append(f"Context:\n{context}" if context else "Context:\n(no context retrieved)")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def _extractive_fallback(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            FALLBACK_LABEL
            + "\n\nNo relevant context was retrieved, so there is nothing "
            "to excerpt. Try rephrasing the question."
        )
    top = chunks[0]
    return f"{FALLBACK_LABEL}\n\nFrom {top.source}, page {top.page}:\n\n{top.text}"


def _call_claude(prompt: str, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
    history: list[dict] | None = None,
    llm_fn: Callable[[str], str] | None = None,
) -> dict:
    """Answer ``question`` from ``chunks``.

    ``llm_fn`` is an injected function (prompt -> answer) used by tests;
    when given, it replaces the real API call. Returns a dict with the
    answer, citations, the provider used, and whether the extractive
    fallback was used.
    """
    citations = citations_for(chunks)
    prompt = build_prompt(question, chunks, history)

    if llm_fn is not None:
        return {
            "answer": llm_fn(prompt),
            "citations": citations,
            "provider": "injected",
            "fallback": False,
        }

    if not settings.has_llm_key:
        return {
            "answer": _extractive_fallback(chunks),
            "citations": citations,
            "provider": "extractive-fallback",
            "fallback": True,
        }

    if settings.llm_provider == "openai":
        answer = _call_openai(prompt, settings.openai_api_key)
    else:
        answer = _call_claude(prompt, settings.anthropic_api_key)
    return {
        "answer": answer,
        "citations": citations,
        "provider": settings.llm_provider,
        "fallback": False,
    }
