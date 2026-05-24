"""Eval runner: loads golden cases, executes the pipeline, computes metrics."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from src.pipeline import RAGPipeline

from .metrics import CaseEvaluator, EvalCase, EvalResult


class EvalRunner:
    def __init__(
        self,
        pipeline: RAGPipeline | None = None,
        evaluator: CaseEvaluator | None = None,
    ) -> None:
        self.pipeline = pipeline or RAGPipeline()
        self.evaluator = evaluator or CaseEvaluator()

    @staticmethod
    def load_cases(path: Path) -> list[EvalCase]:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return [
            EvalCase(
                id=str(item.get("id")),
                category=item.get("category", "lookup"),
                question=item["question"],
                expected_answer=item.get("expected_answer", ""),
                expected_sources=list(item.get("expected_sources", [])),
            )
            for item in raw
        ]

    def run(
        self,
        cases: Iterable[EvalCase],
        *,
        mode: str = "hybrid",
        use_rerank: bool = True,
        on_progress: Callable[[int, int, EvalResult], None] | None = None,
    ) -> list[EvalResult]:
        cases = list(cases)
        results: list[EvalResult] = []
        for i, case in enumerate(cases, start=1):
            response = self.pipeline.ask(
                case.question, mode=mode, use_rerank=use_rerank
            ).to_dict()
            metrics = self.evaluator.evaluate(case, response)
            result = EvalResult(
                case=case,
                answer=response.get("answer", ""),
                citations=response.get("citations", []),
                chunks_used=response.get("chunks", []),
                metrics=metrics,
                raw_response=response,
            )
            results.append(result)
            if on_progress:
                on_progress(i, len(cases), result)
        return results

    @staticmethod
    def aggregate(results: list[EvalResult]) -> dict[str, Any]:
        if not results:
            return {}
        by_category: dict[str, list[EvalResult]] = {}
        for r in results:
            by_category.setdefault(r.case.category, []).append(r)

        def avg(field_name: str, items: list[EvalResult]) -> float:
            values = [getattr(it.metrics, field_name) for it in items]
            return mean(values) if values else 0.0

        agg: dict[str, Any] = {
            "n": len(results),
            "overall": {
                "correctness": avg("correctness", results),
                "faithfulness": avg("faithfulness", results),
                "retrieval_relevance": avg("retrieval_relevance", results),
                "citation_accuracy": avg("citation_accuracy", results),
                "citation_coverage": avg("citation_coverage", results),
                "confidence_composite": avg("confidence_composite", results),
                "idk_match_rate": (
                    sum(
                        1
                        for r in results
                        if r.metrics.is_idk == r.metrics.expected_idk
                    )
                    / len(results)
                ),
            },
            "by_category": {},
        }

        for cat, items in by_category.items():
            agg["by_category"][cat] = {
                "n": len(items),
                "correctness": avg("correctness", items),
                "faithfulness": avg("faithfulness", items),
                "retrieval_relevance": avg("retrieval_relevance", items),
                "citation_accuracy": avg("citation_accuracy", items),
                "citation_coverage": avg("citation_coverage", items),
                "confidence_composite": avg("confidence_composite", items),
                "idk_match_rate": (
                    sum(
                        1
                        for r in items
                        if r.metrics.is_idk == r.metrics.expected_idk
                    )
                    / len(items)
                ),
            }
        return agg
