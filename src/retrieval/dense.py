"""Dense (semantic) retriever backed by Chroma."""
from __future__ import annotations

from typing import Any

from src.embeddings import OpenAIEmbedder
from src.indexing import ChromaVectorStore


class DenseRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        embedder: OpenAIEmbedder | None = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or OpenAIEmbedder()

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        vec = self.embedder.embed_query(query)
        hits = self.vector_store.query(vec, top_k=top_k)
        for rank, hit in enumerate(hits, start=1):
            hit["dense_rank"] = rank
            hit["dense_score"] = hit.get("similarity", 0.0)
            hit["source"] = "dense"
        return hits
