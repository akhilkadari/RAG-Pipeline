"""Smoke tests for each loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.loaders import Document, get_loader
from src.loaders.loader_factory import UnsupportedFileTypeError


@pytest.fixture
def tmp_text(tmp_path: Path) -> Path:
    p = tmp_path / "hello.txt"
    p.write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")
    return p


@pytest.fixture
def tmp_markdown(tmp_path: Path) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(
        "# Title\n\nIntro paragraph.\n\n## Section A\n\nBody of A.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_html(tmp_path: Path) -> Path:
    p = tmp_path / "page.html"
    p.write_text(
        "<html><head><title>Hello</title></head>"
        "<body><h1>Heading</h1><p>Body para.</p>"
        "<script>var leak=1;</script></body></html>",
        encoding="utf-8",
    )
    return p


def test_text_loader_basic(tmp_text: Path) -> None:
    docs = get_loader(tmp_text).load(tmp_text)
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert "Hello world" in doc.text
    assert doc.metadata["filename"] == "hello.txt"
    assert doc.metadata["file_type"] == "txt"
    assert "loaded_at" in doc.metadata


def test_markdown_loader_extracts_sections(tmp_markdown: Path) -> None:
    docs = get_loader(tmp_markdown).load(tmp_markdown)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["top_heading"] == "Title"
    titles = [s["text"] for s in doc.metadata["sections"]]
    assert titles == ["Title", "Section A"]
    assert "Intro paragraph." in doc.text


def test_html_loader_strips_scripts(tmp_html: Path) -> None:
    docs = get_loader(tmp_html).load(tmp_html)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["title"] == "Hello"
    assert "Heading" in doc.metadata["headings"]
    assert "Body para." in doc.text
    assert "leak" not in doc.text


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    bad = tmp_path / "weird.xyz"
    bad.write_text("doesn't matter", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        get_loader(bad)


def test_strict_encoding_fails_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\xfd not utf-8")
    with pytest.raises(UnicodeDecodeError):
        get_loader(bad).load(bad)
