"""ChromaDB-backed vector store wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.chunking import Chunk
from src.config import settings


def _flatten_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only stores str|int|float|bool. Serialize complex values as JSON."""
    flat: dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            try:
                flat[k] = json.dumps(v, ensure_ascii=False)
            except TypeError:
                flat[k] = str(v)
    return flat


def _restore_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
            try:
                out[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        out[k] = v
    return out


class ChromaVectorStore:
    """Persistent Chroma collection sitting under settings.index_dir/chroma."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: Path | None = None,
    ) -> None:
        self.persist_dir = persist_dir or (settings.index_dir / "chroma")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection_name = collection_name or settings.chroma_collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ─── writes ────────────────────────────────────────────
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [_flatten_metadata(c.metadata) for c in chunks]
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=list(embeddings),
            metadatas=metadatas,
        )

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ─── reads ─────────────────────────────────────────────
    def query(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Return ranked hits with id, text, metadata, distance, similarity."""
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0] if result["ids"] else []
        docs = result["documents"][0] if result.get("documents") else []
        metas = result["metadatas"][0] if result.get("metadatas") else []
        dists = result["distances"][0] if result.get("distances") else []

        hits: list[dict[str, Any]] = []
        for i, cid in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            similarity = max(0.0, 1.0 - distance)
            hits.append(
                {
                    "id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": _restore_metadata(metas[i] if i < len(metas) else {}),
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return hits

    def all_chunks(self) -> list[dict[str, Any]]:
        """Pull every chunk back (used to rebuild BM25 etc.)."""
        result = self._collection.get(include=["documents", "metadatas", "embeddings"])
        ids = result.get("ids", []) or []
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        embs = result.get("embeddings", []) or []
        out = []
        for i, cid in enumerate(ids):
            out.append(
                {
                    "id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": _restore_metadata(metas[i] if i < len(metas) else {}),
                    "embedding": list(embs[i]) if i < len(embs) else None,
                }
            )
        return out

    def count(self) -> int:
        return int(self._collection.count())
