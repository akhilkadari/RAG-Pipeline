"""Tests for citation extraction & sentence splitting (no LLM calls)."""
from __future__ import annotations

from src.generation.citation_verifier import _citations_in, _split_sentences
from src.generation.generator import _extract_citations


def test_extract_citations_handles_lists() -> None:
    text = "First claim [1]. Second claim [2, 3]. Third claim [4]."
    assert _extract_citations(text, max_index=4) == [1, 2, 3, 4]


def test_extract_citations_clips_out_of_range() -> None:
    text = "Outdated [99]."
    assert _extract_citations(text, max_index=3) == []


def test_split_sentences_basic() -> None:
    text = "Hello world. Second sentence! And a third? Fourth."
    sents = _split_sentences(text)
    assert len(sents) == 4
    assert sents[0] == "Hello world."


def test_citations_in_handles_multi() -> None:
    sentence = "Mixed claim with multi-cite [1, 2] and trailing [3]."
    assert _citations_in(sentence, max_index=3) == [1, 2, 3]
