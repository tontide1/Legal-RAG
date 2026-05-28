"""Document parsing pipeline for Legal RAG.

PDF files are converted with Docling with OCR disabled. If a PDF has no
extractable text layer, ingestion fails instead of attempting OCR.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


NO_EXTRACTABLE_PDF_TEXT_ERROR = (
    "No extractable text found in PDF. OCR is disabled, so scanned PDFs are not supported."
)


class DocumentProcessor:
    """Async document processor supporting PDF and TXT files."""

    def __init__(self, pdf_converter=None):
        self.pdf_converter = pdf_converter

    async def extract_text(self, file_path: str) -> str:
        """Extract text from a file and return content for LightRAG ingestion."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return await self._extract_pdf(file_path)
        if suffix == ".txt":
            return await self._extract_txt(file_path)
        raise ValueError("Only PDF and TXT files are supported")

    async def _extract_pdf(self, file_path: str) -> str:
        from backend.config import settings

        text = await asyncio.to_thread(
            _convert_pdf_with_docling,
            file_path,
            self.pdf_converter,
        )
        text = text.strip()

        if not text:
            raise ValueError(NO_EXTRACTABLE_PDF_TEXT_ERROR)

        output_dir = Path(settings.LIGHTRAG_WORKING_DIR) / "extracted_txt"
        output_path = output_dir / f"{Path(file_path).stem}.txt"
        await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(output_path.write_text, text, encoding="utf-8")

        print(f"[INFO] Docling extracted {len(text)} chars to {output_path}")
        return text

    async def _extract_txt(self, file_path: str) -> str:
        def read_text() -> str:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")

        return await asyncio.to_thread(read_text)


def _build_docling_pdf_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _convert_pdf_with_docling(file_path: str, converter=None) -> str:
    converter = converter or _build_docling_pdf_converter()

    try:
        result = converter.convert(file_path)
        document = getattr(result, "document", None)
        if document is None:
            raise ValueError("Docling conversion returned no document")

        export_to_markdown = getattr(document, "export_to_markdown", None)
        if not callable(export_to_markdown):
            raise ValueError("Docling document does not support Markdown export")

        return export_to_markdown() or ""
    except ValueError:
        raise
    except Exception as exc:
        filename = Path(file_path).name
        raise ValueError(f"Docling failed to extract PDF text from '{filename}': {exc}") from exc
