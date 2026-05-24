"""LLM-as-judge reranker.

Cross-encoder rerankers (e.g. bge-reranker) are heavy dependencies; using the
generation LLM to score relevance of (query, chunk) pairs gives comparable
quality with no extra deps. The judge is prompted to output a strict 0-10 score.
"""
from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from src.config import settings

_JUDGE_PROMPT = """You are a strict relevance judge for a retrieval system.

Score how relevant the PASSAGE is to the QUESTION on a 0-10 integer scale where:
  - 0  : completely irrelevant
  - 5  : tangentially related
  - 8  : directly relevant, partially answers the question
  - 10 : directly answers the question with strong evidence

Respond with ONLY the integer score. No words, no punctuation, no reasoning.

QUESTION:
{query}

PASSAGE:
{passage}

SCORE:"""

_INT_RE = re.compile(r"\d+")


class LLMReranker:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.judge_model
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError("OPENAI_API_KEY must be set for the LLM reranker.")
        self._client = OpenAI(api_key=key)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        scored: list[dict[str, Any]] = []
        for cand in candidates:
            score = self._judge(query, cand.get("text", ""))
            cand_with_score = {**cand, "rerank_score": score}
            scored.append(cand_with_score)

        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        for i, c in enumerate(scored[:top_k], start=1):
            c["final_rank"] = i
        return scored[:top_k]

    def _judge(self, query: str, passage: str) -> float:
        prompt = _JUDGE_PROMPT.format(query=query, passage=passage[:3000])
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            content = (resp.choices[0].message.content or "").strip()
            match = _INT_RE.search(content)
            if not match:
                return 0.0
            return min(10.0, max(0.0, float(int(match.group(0))))) / 10.0
        except Exception:
            return 0.0
