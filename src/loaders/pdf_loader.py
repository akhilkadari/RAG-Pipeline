from __future__ import annotations
from pathlib import Path
from pypdf import PdfReader
from .base import BaseLoader, Document

class PDFLoader(BaseLoader):
    # the comma is important to make it a tuple with one element
    SUPPORTED_EXTENSIONS = (".pdf",)

    def load(self, path: Path) -> list[Document]:
        # creates a PdfReader object that can be used to read the PDF file
        reader = PdfReader(path)

        # list to store the text of each page
        page_texts : list[str] = []

        # reader.pages is a list of every page in the PDF in order
        # this loops through each page and extracts the text
        for page in reader.pages:
            # tries to pull visible text out of that page
            extracted = page.extract_text() or ""
            # calles _clean_page_text to clean the text
            cleaned = self._clean_page_text(extracted)
            # only add the text if it's not empty
            if cleaned:
                page_texts.append(cleaned)
        
        # joins all pages in page_texts with two newlines between each page
        full_text = "\n\n".join(page_texts)

        # adds basic metadata about the PDF file
        metadata = self._base_metadata(path)
        # number of pages in the PDF
        metadata["page_count"] = len(reader.pages)

        # reader.metadata is a dictionary of the embedded metadata about the PDF file
        info = reader.metadata
        metadata["pdf_title"] = info.get("/Title") if info else None
        metadata["pdf_author"] = info.get("/Author") if info else None


        return [Document(text=full_text, metadata=metadata)]

    def _clean_page_text(self, raw: str) -> str:
        # pdf text extraction is messy 
        # so we clean it by collapsing whitespace and dropping blanks
        lines = [" ".join(line.split()) for line in raw.split("\n")]
        return "\n".join(line for line in lines if line)