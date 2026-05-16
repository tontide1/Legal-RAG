from __future__ import annotations

import asyncio
from pathlib import Path


class DocumentProcessor:
    def __init__(self, ocr_engine=None):
        self.ocr_engine = ocr_engine

    def _create_ocr_engine(self):
        from paddleocr import PaddleOCR

        return PaddleOCR(use_angle_cls=True, lang="en")

    def _get_ocr_engine(self):
        if self.ocr_engine is None:
            self.ocr_engine = self._create_ocr_engine()
        return self.ocr_engine

    async def extract_text(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return await self._extract_pdf(file_path)
        if suffix == ".txt":
            return await self._extract_txt(file_path)
        raise ValueError("Only PDF and TXT files are supported")

    async def _extract_pdf(self, file_path: str) -> str:
        pages = await asyncio.to_thread(_convert_pdf_to_images, file_path)
        ocr_engine = self._get_ocr_engine()

        chunks: list[str] = []
        for page in pages:
            result = await asyncio.to_thread(ocr_engine.ocr, page, cls=True)
            page_text = self._extract_page_text(result)
            if page_text.strip():
                chunks.append(page_text.strip())

        return "\n\n".join(chunks)

    async def _extract_txt(self, file_path: str) -> str:
        def read_text() -> str:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")

        return await asyncio.to_thread(read_text)

    def _extract_page_text(self, result) -> str:
        if not result:
            return ""

        parts: list[str] = []
        for item in result:
            text = self._extract_text_from_ocr_item(item)
            if text.strip():
                parts.append(text.strip())
        return "\n".join(parts)

    def _extract_text_from_ocr_item(self, item) -> str:
        if isinstance(item, str):
            return item

        if isinstance(item, (list, tuple)):
            for value in item:
                text = self._extract_text_from_ocr_item(value)
                if text.strip():
                    return text

        return ""


def _convert_pdf_to_images(file_path: str):
    from pdf2image import convert_from_path

    return convert_from_path(file_path)
