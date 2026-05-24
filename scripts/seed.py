"""One-shot seed: ingest the bundled sample corpus and build indexes.

Idempotent: skips re-indexing if the vector store already has chunks.

Usage:
    python -m scripts.seed
    python -m scripts.seed --force      # rebuild even if chunks already exist
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# Silence ChromaDB telemetry noise (known bug in chromadb==0.5.5).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from src.config import PROJECT_ROOT, settings
from src.indexing import ChromaVectorStore, IndexingPipeline
from src.loaders import Document
from scripts.ingest import ingest_directory


def _load_processed(processed_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(processed_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for item in payload if isinstance(payload, list) else []:
            docs.append(
                Document(
                    text=item.get("text", ""), metadata=item.get("metadata", {})
                )
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the RAG pipeline with the sample corpus.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if the vector store already has data.",
    )
    args = parser.parse_args()

    chroma = ChromaVectorStore()
    if chroma.count() > 0 and not args.force:
        print(f"Vector store already has {chroma.count()} chunks. Skipping seed.")
        print("Use --force to rebuild.")
        return

    sample_dir = PROJECT_ROOT / "data" / "raw" / "sample_corpus"
    if not sample_dir.exists():
        print(f"Sample corpus not found at {sample_dir}. Nothing to seed.")
        return

    print(f"Ingesting sample corpus from {sample_dir} ...")
    ok, skipped, failed = ingest_directory(sample_dir, settings.processed_dir / "sample_corpus")
    print(f"  loaders: ok={ok} skipped={skipped} failed={failed}")

    docs = _load_processed(settings.processed_dir / "sample_corpus")
    print(f"Indexing {len(docs)} documents ...")
    pipeline = IndexingPipeline()
    if args.force:
        pipeline.reset()
        pipeline = IndexingPipeline()
    report = pipeline.index(docs)
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
