"""Image ingestion — Gemini Vision captioning → embeddable text chunks."""
from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import google.generativeai as genai
from langchain_core.documents import Document
from PIL import Image

from config.settings import get_settings

cfg = get_settings()
genai.configure(api_key=cfg.gemini_api_key)

VISION_PROMPT = """\
You are a precise document analyst. Describe this image in detail, covering:
1. Main visual content (objects, people, scenes, charts, diagrams)
2. Any text visible in the image (OCR-style transcription)
3. Key data or insights if it is a chart/graph/table
4. Context or purpose of the image if inferrable

Be thorough — this description will be used for semantic search retrieval.
Format as flowing prose, not bullet points.
"""


@dataclass
class ImageChunk:
    caption: str
    source: str
    doc_hash: str
    width: int
    height: int
    format: str
    doc_type: str = "image"

    def to_langchain_doc(self) -> Document:
        return Document(
            page_content=self.caption,
            metadata={
                "source": self.source,
                "doc_hash": self.doc_hash,
                "doc_type": self.doc_type,
                "width": self.width,
                "height": self.height,
                "format": self.format,
            },
        )


class ImageIngestionPipeline:
    """
    Converts images to searchable text via Gemini Vision (free tier).

    Changes from paid version
    -------------------------
    - OpenAI GPT-4o Vision replaced with google-generativeai Gemini 2.5 Flash
    - Same VISION_PROMPT — Gemini handles the identical instruction format
    - Free tier: 1,500 req/day via Google AI Studio key
    """

    MAX_PIXELS = 1536 * 1536

    def __init__(self) -> None:
        self._model = genai.GenerativeModel(cfg.vision_model)

    def ingest(self, file_path: str | Path) -> ImageChunk:
        path = Path(file_path)
        with open(path, "rb") as f:
            data = f.read()
        return self._process(data, path.name)

    def ingest_bytes(self, data: bytes, filename: str) -> ImageChunk:
        return self._process(data, filename)

    # ── Private ────────────────────────────────────────────────────────

    def _process(self, data: bytes, filename: str) -> ImageChunk:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = self._resize_if_needed(img)
        doc_hash = hashlib.sha256(data).hexdigest()[:16]
        caption = self._caption(img)

        return ImageChunk(
            caption=caption,
            source=filename,
            doc_hash=doc_hash,
            width=img.width,
            height=img.height,
            format="JPEG",
        )

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        if img.width * img.height > self.MAX_PIXELS:
            scale = (self.MAX_PIXELS / (img.width * img.height)) ** 0.5
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)
        return img

    def _caption(self, img: Image.Image) -> str:
        # Gemini accepts PIL Image objects directly
        response = self._model.generate_content([VISION_PROMPT, img])
        return response.text.strip()