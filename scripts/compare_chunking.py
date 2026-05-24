"""Run the eval suite under each chunking strategy and emit a comparison report.

For each strategy in {fixed_size, recursive, semantic}:
  1. Reset the vector store + BM25 index
  2. Re-index processed docs with that chunker
  3. Run the eval suite (hybrid + rerank)
  4. Aggregate metrics

Outputs a JSON report and a Markdown table to data/eval/.

Usage:
    python -m scripts.compare_chunking
    python -m scripts.compare_chunking --strategies fixed_size,recursive
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.chunking import get_chunker
from src.config import PROJECT_ROOT, settings
from src.embeddings import OpenAIEmbedder
from src.eval import EvalRunner
from src.indexing import IndexingPipeline
from src.loaders import Document


def _load_processed_docs(processed_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(processed_dir.rglob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            continue
        for item in payload:
            docs.append(
                Document(
                    text=item.get("text", ""), metadata=item.get("metadata", {})
                )
            )
    return docs


def _markdown_table(results_by_strategy: dict[str, dict]) -> str:
    headers = [
        "strategy",
        "correctness",
        "faithfulness",
        "retrieval_relevance",
        "citation_accuracy",
        "citation_coverage",
        "idk_match",
        "confidence",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for strategy, payload in results_by_strategy.items():
        ov = payload["aggregate"]["overall"]
        row = [
            strategy,
            f"{ov['correctness']:.3f}",
            f"{ov['faithfulness']:.3f}",
            f"{ov['retrieval_relevance']:.3f}",
            f"{ov['citation_accuracy']:.3f}",
            f"{ov['citation_coverage']:.3f}",
            f"{ov['idk_match_rate']:.3f}",
            f"{ov['confidence_composite']:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies on the eval suite.")
    parser.add_argument(
        "--strategies",
        type=str,
        default="fixed_size,recursive,semantic",
        help="Comma-separated list of chunkers to compare.",
    )
    parser.add_argument(
        "--processed", type=Path, default=settings.processed_dir
    )
    parser.add_argument(
        "--golden", type=Path, default=PROJECT_ROOT / "data" / "eval" / "golden.json"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=PROJECT_ROOT / "data" / "eval"
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    docs = _load_processed_docs(args.processed)
    if not docs:
        print(
            f"No processed documents in {args.processed}. "
            "Run `python -m scripts.ingest` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    embedder = OpenAIEmbedder()
    runner = EvalRunner()
    cases = runner.load_cases(args.golden)
    print(f"Comparing {len(strategies)} strategies on {len(cases)} cases...\n")

    by_strategy: dict[str, dict] = {}

    for strategy in strategies:
        print(f"=== Strategy: {strategy} ===")
        chunker = get_chunker(
            strategy,  # type: ignore[arg-type]
            embed_fn=embedder.embed if strategy == "semantic" else None,
        )
        pipeline = IndexingPipeline(chunker=chunker, embedder=embedder)
        pipeline.reset()
        pipeline = IndexingPipeline(chunker=chunker, embedder=embedder)
        report = pipeline.index(docs)
        print(f"Indexed: {report.chunks_indexed} chunks")

        # Re-instantiate the eval runner so the retriever picks up the new index.
        runner = EvalRunner()

        def progress(i: int, n: int, result) -> None:
            print(
                f"  [{i:>3}/{n}] {result.case.id} correct={result.metrics.correctness:.2f}",
                end="\r",
            )

        results = runner.run(cases, mode="hybrid", use_rerank=True, on_progress=progress)
        print()
        aggregate = runner.aggregate(results)
        by_strategy[strategy] = {
            "indexing": report.as_dict(),
            "aggregate": aggregate,
            "results": [r.as_dict() for r in results],
        }
        print(json.dumps(aggregate["overall"], indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "chunking_comparison.json"
    md_path = args.out_dir / "chunking_comparison.md"
    json_path.write_text(json.dumps(by_strategy, indent=2, ensure_ascii=False))

    md = "# Chunking Strategy Comparison\n\n" + _markdown_table(by_strategy)
    md_path.write_text(md)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print("\n" + md)


if __name__ == "__main__":
    main()
