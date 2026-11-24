"""Local embeddings via sentence-transformers.

The heavy import (sentence_transformers -> torch) happens inside __init__,
never at module top level, so importing this module never loads torch —
tests and the extractive-fallback path stay light.
"""
from collections.abc import Callable


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        # First run downloads the model (~90 MB) to a local cache.
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,  # cosine similarity behaves correctly
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


EmbedFn = Callable[[str], list[float]]
