from __future__ import annotations
from pathlib import Path
from markdown_it import MarkdownIt
from .base import BaseLoader, Document

class MarkdownLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = (".md", ".markdown")
    
    def __init__(self) -> None:
        # Markdownit is the parser; constructing it once per loader is efficient.
        self._md = MarkdownIt()
    
    def load(self, path: Path) -> list[Document]:
        # reads the file into a string raw. 
        # encoding="utf-8" is the default encoding for standard .md files. (bytes to text)
        raw = path.read_text(encoding="utf-8")
        #parse returns a list of token objects
        # of which each token has things like type, tag, headings, content, etc.
        tokens = self._md.parse(raw)

        # stores small dicts each describing one heading and text
        sections: list[dict[str, object]] = []
        # lines of plain text you join into the final Document.text
        text_lines: list[str] = []

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                # tok.tag is 'h1', 'h2', etc.
                level = int(tok.tag[1])
                inline = tokens[i+1]
                heading_text = inline.content.strip()
                sections.append({"level": level, "text": heading_text})
                # keep heading visible in the plaintext so LLM has context
                text_lines.append(f"{'#' * level} {heading_text}")
                i += 3 # skip heading_open, inline, heading_close
                continue
            if tok.type == "inline":
                # body paragraphs, list items, etc.
                text_lines.append(tok.content.strip())
            i += 1

        text = "\n\n".join(line for line in text_lines if line)

        metadata = self._base_metadata(path)
        metadata["sections"] = sections
        metadata["top_heading"] = sections[0]["text"] if sections else None

        return [Document(text=text, metadata=metadata)]

