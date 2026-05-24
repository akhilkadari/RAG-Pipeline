"""End-to-end indexing pipeline: Documents -> Chunks -> Dedup -> Embed -> Indexes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.chunking import BaseChunker, Chunk, get_chunker
from src.config import settings
from src.embeddings import OpenAIEmbedder
from src.loaders import Document

from .bm25_index import BM25Index
from .deduplicator import Deduplicator
from .vector_store import ChromaVectorStore


@dataclass
class IndexingReport:
    documents_in: int = 0
    chunks_produced: int = 0
    chunks_indexed: int = 0
    duplicates_dropped: int = 0
    chunker: str = ""
    drops: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "documents_in": self.documents_in,
            "chunks_produced": self.chunks_produced,
            "chunks_indexed": self.chunks_indexed,
            "duplicates_dropped": self.duplicates_dropped,
            "chunker": self.chunker,
            "drops": self.drops,
        }


class IndexingPipeline:
    """Orchestrates chunking + embedding + dedup + dual-index storage.

    The vector store and BM25 index stay in sync — we only insert into BM25
    chunks that survived dedup and were inserted into Chroma.
    """

    def __init__(
        self,
        chunker: BaseChunker | None = None,
        embedder: OpenAIEmbedder | None = None,
        vector_store: ChromaVectorStore | None = None,
        bm25_index: BM25Index | None = None,
        deduplicator: Deduplicator | None = None,
    ) -> None:
        self.embedder = embedder or OpenAIEmbedder()
        self.chunker = chunker or get_chunker(
            settings.default_chunker,  # type: ignore[arg-type]
            embed_fn=self.embedder.embed if settings.default_chunker == "semantic" else None,
        )
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.deduplicator = deduplicator or Deduplicator()

        # Bootstrap dedup against whatever already lives in the vector store.
        existing = self.vector_store.all_chunks()
        if existing:
            seed_chunks = [
                Chunk(id=e["id"], text=e["text"], metadata=e["metadata"])
                for e in existing
                if e.get("embedding") is not None
            ]
            seed_embs = [e["embedding"] for e in existing if e.get("embedding") is not None]
            self.deduplicator.seed(seed_chunks, seed_embs)
            # And rebuild BM25 from existing chunks if disk copy is missing.
            if not self.bm25_index.load():
                self.bm25_index.build(
                    ids=[e["id"] for e in existing],
                    texts=[e["text"] for e in existing],
                    metadatas=[e["metadata"] for e in existing],
                )
                self.bm25_index.save()

    def index(self, documents: Sequence[Document]) -> IndexingReport:
        report = IndexingReport(documents_in=len(documents), chunker=self.chunker.name)
        if not documents:
            return report

        chunks = self.chunker.split(list(documents))
        report.chunks_produced = len(chunks)
        if not chunks:
            return report

        embeddings = self.embedder.embed([c.text for c in chunks])
        kept_chunks, kept_embs, drops = self.deduplicator.filter(chunks, embeddings)
        report.duplicates_dropped = len(drops)
        report.drops = drops

        if not kept_chunks:
            return report

        self.vector_store.add(kept_chunks, kept_embs)

        # Rebuild BM25 over the union of existing + new chunks.
        all_chunks = self.vector_store.all_chunks()
        self.bm25_index.build(
            ids=[c["id"] for c in all_chunks],
            texts=[c["text"] for c in all_chunks],
            metadatas=[c["metadata"] for c in all_chunks],
        )
        self.bm25_index.save()

        report.chunks_indexed = len(kept_chunks)
        return report

    def reset(self) -> None:
        """Clear the vector store and BM25 index. Useful for re-indexing."""
        self.vector_store.reset()
        self.bm25_index = BM25Index()
        if self.bm25_index.persist_path.exists():
            self.bm25_index.persist_path.unlink()
        self.deduplicator = Deduplicator()
