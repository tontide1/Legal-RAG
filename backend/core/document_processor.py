"""document_processor.py – Document parsing pipeline for Legal RAG.

Pipeline decision tree:
    PDF
    ├─ Has text layer (≥ MIN_DIRECT_TEXT_LEN chars)?
    │   └─ YES → pymupdf4llm → Markdown  (fast, structure-preserving)
    └─ NO (scan / image-only)
        └─ pdf2image → PaddleOCR (PP-OCRv5 / standard)
            ├─ confidence OK? → ocr_to_markdown → done
            └─ confidence LOW → fallback: Chandra OCR 2

    TXT  → read directly

OCR confidence threshold:
    If the average confidence across all detected text lines is below
    OCR_CONFIDENCE_THRESHOLD the result is considered unreliable and the
    Chandra-OCR fallback is triggered.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

# Minimum characters extracted directly from PDF to skip OCR
_MIN_DIRECT_TEXT_LEN = 100

# Minimum average OCR confidence (0–1) to trust PaddleOCR output
OCR_CONFIDENCE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Public processor
# ---------------------------------------------------------------------------


class DocumentProcessor:
    """Async document processor supporting PDF (text + scan) and TXT."""

    def __init__(self, ocr_engine=None):
        self.ocr_engine = ocr_engine

    # ------------------------------------------------------------------
    # OCR engine lazy init
    # ------------------------------------------------------------------

    def _create_ocr_engine(self):
        """Instantiate PaddleOCR (PP-OCRv5 if available, else standard v3)."""
        from paddleocr import PaddleOCR

        try:
            # PP-OCRv5 – higher accuracy, requires paddleocr >= 3.x
            engine = PaddleOCR(
                ocr_version="PP-OCRv5",
                use_angle_cls=True,
                lang="vi",
                show_log=False,
            )
            print("[INFO] PaddleOCR PP-OCRv5 initialised.")
            return engine
        except Exception:
            # Fallback to standard PP-OCRv3 (always available)
            engine = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
            print("[INFO] PaddleOCR standard (PP-OCRv3) initialised.")
            return engine

    def _get_ocr_engine(self):
        if self.ocr_engine is None:
            self.ocr_engine = self._create_ocr_engine()
        return self.ocr_engine

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def extract_text(self, file_path: str) -> str:
        """Extract text from a file and return Markdown-formatted content.

        Returned string is suitable for :func:`legal_chunker.chunk_markdown`.
        """
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return await self._extract_pdf(file_path)
        if suffix == ".txt":
            return await self._extract_txt(file_path)
        raise ValueError("Only PDF and TXT files are supported")

    # ------------------------------------------------------------------
    # PDF branch
    # ------------------------------------------------------------------

    async def _extract_pdf(self, file_path: str) -> str:
        from backend.config import settings

        # --- Branch 1: Text-based PDF → pymupdf4llm (Markdown, fast) ---
        try:
            markdown_text = await asyncio.to_thread(
                _extract_pdf_text_pymupdf4llm, file_path
            )
        except Exception as exc:
            print(f"[WARN] pymupdf4llm failed ({exc}), attempting pypdf fallback.")
            markdown_text = await asyncio.to_thread(
                _extract_pdf_text_pypdf, file_path
            )

        if len(markdown_text.strip()) >= _MIN_DIRECT_TEXT_LEN:
            print(
                f"[INFO] Text-layer extraction succeeded for {file_path} "
                f"({len(markdown_text)} chars)"
            )
            return markdown_text

        # --- Branch 2: Scan PDF / image-only → PaddleOCR ---
        print(
            f"[INFO] Direct extraction insufficient "
            f"({len(markdown_text.strip())} chars), falling back to OCR for {file_path}"
        )

        base_name = Path(file_path).stem
        images_dir = Path(settings.LIGHTRAG_WORKING_DIR) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        pages = await asyncio.to_thread(
            _convert_pdf_to_images, file_path, settings.POPPLER_PATH
        )
        ocr_engine = self._get_ocr_engine()

        page_results: list[tuple[str, float]] = []  # (text, avg_confidence)

        for i, page in enumerate(pages, 1):
            img_path = images_dir / f"{base_name}_page_{i}.png"
            await asyncio.to_thread(page.save, str(img_path), "PNG")

            raw = await asyncio.to_thread(
                lambda p=img_path: ocr_engine.ocr(str(p), cls=True)
            )
            page_text, avg_conf = _extract_page_text_with_confidence(raw)
            if page_text.strip():
                page_results.append((page_text.strip(), avg_conf))

        if not page_results:
            return ""

        overall_confidence = sum(c for _, c in page_results) / len(page_results)

        # --- Branch 2a: Confidence OK → convert OCR lines to Markdown ---
        if overall_confidence >= OCR_CONFIDENCE_THRESHOLD:
            print(
                f"[INFO] OCR confidence {overall_confidence:.2f} ≥ threshold "
                f"({OCR_CONFIDENCE_THRESHOLD}) — using PaddleOCR output."
            )
            return _ocr_pages_to_markdown([t for t, _ in page_results])

        # --- Branch 2b: Confidence LOW → Chandra OCR 2 fallback ---
        print(
            f"[WARN] OCR confidence {overall_confidence:.2f} < threshold "
            f"({OCR_CONFIDENCE_THRESHOLD}) — attempting Chandra OCR 2 fallback."
        )
        try:
            chandra_text = await asyncio.to_thread(
                _chandra_ocr_fallback, file_path
            )
            if chandra_text.strip():
                print("[INFO] Chandra OCR 2 fallback succeeded.")
                return chandra_text
        except Exception as exc:
            print(f"[WARN] Chandra OCR 2 fallback failed: {exc}")

        # Last resort: return whatever PaddleOCR produced despite low confidence
        print("[WARN] Using low-confidence PaddleOCR output as last resort.")
        return _ocr_pages_to_markdown([t for t, _ in page_results])

    # ------------------------------------------------------------------
    # TXT branch
    # ------------------------------------------------------------------

    async def _extract_txt(self, file_path: str) -> str:
        def read_text() -> str:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")

        return await asyncio.to_thread(read_text)


# ---------------------------------------------------------------------------
# Module-level helpers (run in thread pool via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _extract_pdf_text_pymupdf4llm(file_path: str) -> str:
    """Extract text from a digital PDF as Markdown using pymupdf4llm.

    Preserves headers, bold/italic, lists — ideal for legal structure detection.
    Returns empty string for scan-only PDFs.
    """
    import pymupdf4llm  # type: ignore

    md_text = pymupdf4llm.to_markdown(file_path)
    return md_text or ""


def _extract_pdf_text_pypdf(file_path: str) -> str:
    """Fallback plain-text extraction via pypdf (no Markdown)."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def _extract_page_text_with_confidence(result) -> tuple[str, float]:
    """Parse PaddleOCR result and return (text, average_confidence).

    PaddleOCR result structure:
        list[page] where page = list[line]
        line = [[box_coords], (text, confidence)]
    """
    if not result:
        return "", 0.0

    parts: list[str] = []
    confidences: list[float] = []

    for page in result:
        if not page:
            continue
        for line in page:
            if not line or len(line) < 2:
                continue
            text_info = line[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text, conf = text_info[0], text_info[1]
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    if isinstance(conf, (int, float)):
                        confidences.append(float(conf))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return "\n".join(parts), avg_conf


def _ocr_pages_to_markdown(pages: list[str]) -> str:
    """Convert list of per-page OCR text strings into simple Markdown.

    Adds a page separator comment so the legal chunker can later use it
    as a soft hint for citation (page number in metadata).
    """
    parts: list[str] = []
    for i, page_text in enumerate(pages, 1):
        parts.append(f"<!-- page {i} -->\n{page_text}")
    return "\n\n".join(parts)


def _chandra_ocr_fallback(file_path: str) -> str:
    """Chandra OCR 2 fallback for low-confidence scans.

    Chandra OCR 2 is expected to be installed in the same Python environment.
    Adjust the import path / API call to match the actual installed package.
    If the package is not installed, this raises ImportError which the caller
    catches and logs gracefully.
    """
    try:
        # Primary import path for Chandra OCR 2 (adjust if package name differs)
        from chandra_ocr import ChandraOCR  # type: ignore
    except ImportError:
        raise ImportError(
            "chandra_ocr package not found. Install it or remove the fallback reference."
        )

    ocr = ChandraOCR()
    result = ocr.process(file_path)
    if isinstance(result, str):
        return result
    # Some Chandra builds return a dict with a 'text' key
    if isinstance(result, dict):
        return result.get("text", "")
    return ""


def _convert_pdf_to_images(file_path: str, poppler_path: str | None = None):
    """Convert PDF pages to PIL images using pdf2image."""
    from pdf2image import convert_from_path

    kwargs: dict = {}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    return convert_from_path(file_path, **kwargs)
