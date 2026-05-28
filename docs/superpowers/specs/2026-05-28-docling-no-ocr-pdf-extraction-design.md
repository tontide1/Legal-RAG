# Docling PDF Extraction Without OCR Design

## Goal

Replace the current PDF extraction pipeline with Docling and remove OCR from the backend ingestion path entirely.

The system must:

- Use Docling for PDF-to-text extraction.
- Configure Docling with OCR disabled.
- Save an extracted `.txt` artifact for each uploaded PDF.
- Use the saved text content for the existing LightRAG indexing flow.
- Fail clearly for scanned/image-only PDFs that do not contain extractable text.
- Remove runtime OCR code paths and user-facing documentation that describe OCR as part of ingestion.

## Non-Goals

- No PaddleOCR fallback.
- No Chandra OCR fallback.
- No image conversion for OCR.
- No background ingestion queue.
- No frontend behavior change beyond surfacing the backend upload error that already exists.
- No change to LightRAG normalization or insert semantics.
- No migration or deletion of historical OCR output files already under `data/`.

## Current Flow

`POST /api/upload` stores the uploaded file temporarily, calls `DocumentProcessor.extract_text(file_path)`, normalizes the returned content with `normalize_for_lightrag()`, and inserts it with:

```python
await rag.ainsert(
    normalized_content,
    file_paths=[filename],
    split_by_character="\n\n",
)
```

This route contract should remain unchanged. The implementation change belongs behind `DocumentProcessor.extract_text()`.

## Proposed Architecture

Use the existing `DocumentProcessor` boundary.

For `.txt`:

- Keep current behavior: read UTF-8 text directly and return it.
- Do not create a second artifact for user-uploaded `.txt`; the original upload is already text.

For `.pdf`:

- Convert with Docling's Python API using `DocumentConverter`.
- Configure `PdfPipelineOptions(do_ocr=False)`.
- Export the converted document to a text-like representation suitable for the existing legal normalizer.
- Save the extracted output to `settings.LIGHTRAG_WORKING_DIR/extracted_txt/<pdf_stem>.txt`.
- Return that same extracted text to the upload route.

Docling API shape verified from current docs:

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
)
result = converter.convert(file_path)
text = result.document.export_to_markdown()
```

Markdown export is acceptable because the current downstream normalizer already expects either plain text or Markdown and detects Vietnamese legal structure from text markers such as `Điều`.

## Error Handling

If Docling conversion fails:

- Raise a `ValueError` with a concise message identifying PDF extraction failure.
- The existing upload route will wrap it as `HTTPException(500, "Failed to index file: ...")`.

If Docling conversion succeeds but produces empty text:

- Raise a clear error:

```text
No extractable text found in PDF. OCR is disabled, so scanned PDFs are not supported.
```

This makes the no-OCR behavior explicit and prevents silent indexing of empty documents.

## Dependency Changes

Add:

- `docling`

Remove from backend runtime dependencies:

- `pymupdf4llm`
- `pypdf`
- `pdf2image`
- `paddleocr`

Remove Docker-only PaddlePaddle installation from `backend/Dockerfile`.

If any remaining backend helper imports these packages only for legacy OCR or legacy PDF parsing, remove that helper rather than keeping the dependency alive.

## Files To Change

- `backend/core/document_processor.py`
- `backend/core/llm_services.py` if legacy PDF/OCR helper code still imports removed PDF/OCR dependencies
- `backend/requirements.txt`
- `backend/Dockerfile`
- `backend/tests/test_document_processor.py`
- `README.md`
- `ARCHITECTURE.md`
- `PROJECT_STRUCTURE.md`

Serena memories should be updated after implementation because current memories still describe OCR and old PDF extraction.

## Test Plan

Backend unit tests:

- TXT upload extraction still reads UTF-8 file content.
- PDF extraction calls Docling with OCR disabled.
- PDF extraction writes `extracted_txt/<stem>.txt`.
- Empty Docling output raises the no-OCR scanned-PDF error.
- Runtime backend no longer imports `paddleocr`, `pdf2image`, `pymupdf4llm`, or `pypdf`.

Existing upload route tests should continue to pass because `DocumentProcessor.extract_text()` still returns a string.

Verification command:

```bash
pytest backend/tests
```

## Risks

Docling may introduce heavier install/runtime dependencies than the previous text-layer-only path. This is acceptable for the requested simplification, but Docker build time may increase.

PDFs without embedded text will no longer be indexable. This is intentional and should be visible in upload errors.

Markdown export can include table/layout syntax. This should be compatible with the current legal normalizer, but representative Vietnamese legal PDFs should be tested manually after implementation.
