"""Configuration for rag-chatbot-fastapi, read from environment / .env."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "claude")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    collection: str = os.getenv("COLLECTION", "my-docs")
    top_k: int = int(os.getenv("TOP_K", "5"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "10"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    history_db: str = os.getenv("HISTORY_DB", "./history.db")

    @property
    def llm_api_key(self) -> str:
        """The API key for the configured provider, or '' if unset."""
        if self.llm_provider == "openai":
            return self.openai_api_key
        return self.anthropic_api_key

    @property
    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key)


def get_settings() -> Settings:
    return Settings()
