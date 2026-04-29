"""HTML loader using BeautifulSoup."""
from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from .base import BaseLoader, Document


class HTMLLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = (".html", ".htm")

    # Tags whose contents are not human-readable text.
    _NOISE_TAGS = ("script", "style", "noscript", "template", "svg")

    # Collapse 3 or more newlines down to exactly 2 (paragraph break).
    _MULTI_NEWLINE = re.compile(r"\n{3,}")

    def load(self, path: Path) -> list[Document]:
        raw = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw, "lxml")

        # Remove noise tags so their contents don't leak into the plaintext.
        for tag in soup(list(self._NOISE_TAGS)):
            tag.decompose()

        title = (
            soup.title.string.strip()
            if soup.title and soup.title.string
            else None
        )
        headings = [
            h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])
        ]

        # separator='\n' keeps paragraph boundaries readable
        text = soup.get_text(separator="\n", strip=True)
        text = self._MULTI_NEWLINE.sub("\n\n", text)

        metadata = self._base_metadata(path)
        metadata["title"] = title
        metadata["headings"] = headings

        return [Document(text=text, metadata=metadata)]