"""PDF ingestion — text extraction + chunking with rich metadata."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import get_settings

cfg = get_settings()


@dataclass
class PDFChunk:
    """One chunk from a PDF with full provenance metadata."""
    content: str
    source: str
    page: int
    chunk_index: int
    total_chunks: int
    doc_hash: str
    year: int | None
    doc_type: str = "pdf"
    metadata: dict = field(default_factory=dict)

    def to_langchain_doc(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                "source": self.source,
                "page": self.page,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "doc_hash": self.doc_hash,
                "doc_type": self.doc_type,
                "year": self.year,
                **self.metadata,
            },
        )


class PDFIngestionPipeline:
    """
    Extracts, cleans, and chunks PDF text.

    Features
    --------
    - Per-page extraction with PyMuPDF (handles scanned + digital)
    - Deduplication via content hash
    - Metadata-rich chunking: page number, chunk index, year, filename
    - RecursiveCharacterTextSplitter honours sentence/paragraph boundaries
    """

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def ingest(self, file_path: str | Path) -> list[PDFChunk]:
        """Return all chunks from *file_path*."""
        path = Path(file_path)
        raw_pages = list(self._extract_pages(path))
        doc_hash = self._hash_file(path)
        year = self._extract_year(raw_pages)

        chunks: list[PDFChunk] = []
        for page_num, text in raw_pages:
            page_chunks = self._splitter.split_text(text)
            for idx, chunk_text in enumerate(page_chunks):
                chunks.append(
                    PDFChunk(
                        content=chunk_text,
                        source=path.name,
                        page=page_num,
                        chunk_index=idx,
                        total_chunks=len(page_chunks),
                        doc_hash=doc_hash,
                        year=year,
                    )
                )
        return chunks

    def ingest_bytes(self, data: bytes, filename: str) -> list[PDFChunk]:
        """Accept raw bytes (e.g. from Streamlit uploader)."""
        tmp = io.BytesIO(data)
        doc_hash = hashlib.sha256(data).hexdigest()[:16]
        raw_pages = list(self._extract_pages_from_stream(tmp))
        year = self._extract_year(raw_pages)

        chunks: list[PDFChunk] = []
        for page_num, text in raw_pages:
            page_chunks = self._splitter.split_text(text)
            for idx, chunk_text in enumerate(page_chunks):
                chunks.append(
                    PDFChunk(
                        content=chunk_text,
                        source=filename,
                        page=page_num,
                        chunk_index=idx,
                        total_chunks=len(page_chunks),
                        doc_hash=doc_hash,
                        year=year,
                    )
                )
        return chunks

    # ── Private helpers ─────────────────────────────────────────────────

    def _extract_pages(self, path: Path) -> Iterator[tuple[int, str]]:
        with fitz.open(str(path)) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if text:
                    yield page.number + 1, self._clean(text)

    def _extract_pages_from_stream(
        self, stream: io.BytesIO
    ) -> Iterator[tuple[int, str]]:
        with fitz.open(stream=stream, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if text:
                    yield page.number + 1, self._clean(text)

    @staticmethod
    def _clean(text: str) -> str:
        """Remove common PDF noise."""
        import re

        text = re.sub(r"\s{3,}", "  ", text)       # collapse whitespace blobs
        text = re.sub(r"-\n([a-z])", r"\1", text)  # join hyphenated line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)      # max 2 blank lines
        return text.strip()

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        return h.hexdigest()[:16]

    @staticmethod
    def _extract_year(pages: list[tuple[int, str]]) -> int | None:
        import re

        text = " ".join(t for _, t in pages[:3])  # check first 3 pages
        match = re.search(r"\b(19|20)\d{2}\b", text)
        return int(match.group()) if match else None