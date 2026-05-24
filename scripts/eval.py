"""Run the eval suite against the currently-indexed corpus.

Usage:
    python -m scripts.eval
    python -m scripts.eval --mode dense          # ablation: dense-only
    python -m scripts.eval --mode sparse
    python -m scripts.eval --no-rerank
    python -m scripts.eval --golden custom.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.eval import EvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG eval suite.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "golden.json",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
    )
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "last_run.json",
    )
    args = parser.parse_args()

    runner = EvalRunner()
    cases = runner.load_cases(args.golden)
    print(f"Running {len(cases)} cases (mode={args.mode}, rerank={not args.no_rerank})...")

    def progress(i: int, n: int, result) -> None:
        m = result.metrics
        print(
            f"[{i:>3}/{n}] {result.case.id} {result.case.category:>10s} "
            f"correct={m.correctness:.2f} faith={m.faithfulness:.2f} "
            f"retr={m.retrieval_relevance:.2f} cit_acc={m.citation_accuracy:.2f}"
        )

    results = runner.run(
        cases,
        mode=args.mode,
        use_rerank=not args.no_rerank,
        on_progress=progress,
    )

    aggregate = runner.aggregate(results)
    payload = {
        "mode": args.mode,
        "use_rerank": not args.no_rerank,
        "aggregate": aggregate,
        "results": [r.as_dict() for r in results],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nResults written to {args.out}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
