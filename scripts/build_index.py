"""Read processed JSON Documents, chunk them, embed, dedup, store in indexes.

Usage:
    python -m scripts.build_index                       # default chunker
    python -m scripts.build_index --chunker semantic    # override
    python -m scripts.build_index --reset               # drop existing index first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.chunking import get_chunker
from src.config import settings
from src.embeddings import OpenAIEmbedder
from src.indexing import IndexingPipeline
from src.loaders import Document


def _load_documents(processed_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(processed_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            print(f"WARN unexpected JSON shape in {path}, skipping")
            continue
        for item in payload:
            docs.append(
                Document(
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}),
                )
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk + embed + store processed docs.")
    parser.add_argument(
        "--processed", type=Path, default=settings.processed_dir
    )
    parser.add_argument(
        "--chunker", choices=["fixed_size", "recursive", "semantic"], default=None
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing vector store + BM25 index before indexing.",
    )
    args = parser.parse_args()

    if not args.processed.exists():
        print(
            f"Processed directory does not exist: {args.processed}\n"
            "Run `python -m scripts.ingest` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    docs = _load_documents(args.processed)
    if not docs:
        print(f"No processed documents found in {args.processed}.")
        sys.exit(0)

    embedder = OpenAIEmbedder()
    chunker_name = args.chunker or settings.default_chunker
    chunker = get_chunker(
        chunker_name,  # type: ignore[arg-type]
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embed_fn=embedder.embed if chunker_name == "semantic" else None,
    )

    pipeline = IndexingPipeline(chunker=chunker, embedder=embedder)

    if args.reset:
        print("Resetting vector store and BM25 index...")
        pipeline.reset()
        # Re-bootstrap a clean pipeline (now with empty stores).
        pipeline = IndexingPipeline(chunker=chunker, embedder=embedder)

    print(f"Indexing {len(docs)} document(s) using chunker={chunker_name}...")
    report = pipeline.index(docs)
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
