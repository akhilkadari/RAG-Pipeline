"""Thin OpenAI embeddings wrapper with batching and retry logic."""
from __future__ import annotations

import time
from typing import Sequence

from openai import OpenAI

from src.config import settings


class OpenAIEmbedder:
    """Embed text with `text-embedding-3-small` (or whatever EMBEDDING_MODEL is set to)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 64,
        max_retries: int = 4,
    ) -> None:
        self.model = model or settings.embedding_model
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = OpenAI(api_key=key)
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of strings; returns a list of vectors aligned with input order."""
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            results.extend(self._embed_batch(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # ─── internals ─────────────────────────────────────────

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                resp = self._client.embeddings.create(model=self.model, input=batch)
                return [item.embedding for item in resp.data]
            except Exception:
                attempt += 1
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(2**attempt, 8))
