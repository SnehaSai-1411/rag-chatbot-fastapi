"""Streamlit UI for the RAG chatbot.

Talks to the FastAPI backend (API_URL, default http://localhost:8000).
Run it with:  streamlit run ui/app.py
"""
import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot")
st.caption("Ask a question about the sample docs — answers come with citations.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.session_id, language=None)
    if st.button("New session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

question = st.text_input("Your question", placeholder="What is chunking?")
ask = st.button("Ask")

if ask and question.strip():
    with st.spinner("Retrieving and answering…"):
        try:
            response = httpx.post(
                f"{API_URL}/ask",
                json={"question": question, "session_id": st.session_state.session_id},
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            st.stop()
    result = response.json()
    if result.get("fallback"):
        st.warning(
            "No LLM API key is configured, so this is an extractive excerpt "
            "from the retrieved context — not a generated answer."
        )
    st.markdown("### Answer")
    st.write(result["answer"])
    if result.get("citations"):
        st.markdown("### Citations")
        for cite in result["citations"]:
            st.markdown(f"- **{cite['source']}**, page {cite['page']}")

st.divider()
st.markdown("### Session history")
try:
    history = httpx.get(
        f"{API_URL}/sessions/{st.session_state.session_id}/history", timeout=10
    ).json()
    for turn in history.get("turns", []):
        role = "You" if turn["role"] == "user" else "Assistant"
        st.markdown(f"**{role}:** {turn['content']}")
except httpx.HTTPError:
    st.caption("History unavailable — is the API running?")
