"""Prompt templates used by the generator and the judges."""
from __future__ import annotations

GROUNDED_SYSTEM_PROMPT = """You are a careful technical assistant for an internal documentation Q&A system.

You MUST follow these rules:
1. Answer ONLY using the numbered CONTEXT BLOCKS the user provides. Do NOT use outside knowledge.
2. After every factual sentence, cite the supporting context with bracketed numbers like [1] or [2, 3]. Citations refer to the block numbers.
3. If the context does NOT contain enough information to answer the question, you MUST respond with exactly:
   I don't have enough information in the provided documentation to answer that.
   followed by a one-sentence summary of what *is* covered in the context.
4. Never invent file names, page numbers, function names, or quoted text that isn't in the context.
5. Be concise. Prefer bullet points for multi-part answers.
6. If two context blocks disagree, note the disagreement and cite both.
"""

USER_TEMPLATE = """QUESTION:
{question}

CONTEXT BLOCKS (numbered):
{context}

Write your answer now. Remember to cite blocks with bracketed numbers."""


def format_context(chunks: list[dict]) -> str:
    """Render chunks as numbered context blocks."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("filename") or meta.get("source") or "unknown"
        section = meta.get("section_heading")
        header_bits = [f"[{i}] source={source}"]
        if section:
            header_bits.append(f"section={section}")
        parts.append(" | ".join(header_bits) + "\n" + (chunk.get("text") or "").strip())
    return "\n\n---\n\n".join(parts)


CITATION_JUDGE_PROMPT = """You verify whether a CITATION supports a CLAIM.

Output a strict JSON object with two keys:
  "supported": true | false
  "reason": short string (under 25 words)

A citation supports a claim when the cited passage directly states or strongly implies the claim.
Do NOT mark as supported if the passage is only loosely related.

CLAIM:
{claim}

CITED PASSAGE:
{passage}

Respond with JSON only."""


COMPLETENESS_PROMPT = """You evaluate whether an ANSWER addresses every distinct part of a QUESTION.

Output a strict JSON object with two keys:
  "score": float between 0.0 and 1.0  (1.0 = answers all parts; 0.0 = answers none)
  "missing_parts": string list (each item is a question part the answer ignored)

QUESTION:
{question}

ANSWER:
{answer}

Respond with JSON only."""


CORRECTNESS_PROMPT = """You compare a SYSTEM ANSWER against a GOLDEN ANSWER for a QUESTION.

Output a strict JSON object with two keys:
  "score": float between 0.0 and 1.0  (1.0 = semantically equivalent; 0.0 = wrong or contradictory)
  "reason": short string (under 30 words)

QUESTION:
{question}

GOLDEN ANSWER:
{golden}

SYSTEM ANSWER:
{system}

Respond with JSON only."""


FAITHFULNESS_PROMPT = """You verify whether every claim in an ANSWER is supported by the provided CONTEXT.

Output a strict JSON object with two keys:
  "score": float between 0.0 and 1.0  (1.0 = every claim grounded; 0.0 = none grounded / hallucinated)
  "unsupported_claims": list of strings (claims not supported by context)

CONTEXT:
{context}

ANSWER:
{answer}

Respond with JSON only."""
