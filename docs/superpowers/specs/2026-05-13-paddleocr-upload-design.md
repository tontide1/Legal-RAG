# PaddleOCR Upload Pipeline Design

## Goal

Replace the current PDF parsing path in `/api/upload` with a local pipeline:

- PDF upload -> `pdf2image` -> PaddleOCR -> extracted text
- TXT upload -> read file -> text
- extracted text -> `rag.ainsert()` -> document status tracking -> temp file cleanup

This change removes the existing Qwen VL dependency from upload processing.

## Scope

In scope:

- Update the upload endpoint to process PDFs with `pdf2image` and PaddleOCR.
- Keep TXT uploads working through direct file reads.
- Ensure OCR runs through `asyncio.to_thread` so blocking work does not freeze the event loop.
- Fail the whole request if any page extraction step fails.
- Always delete the temporary uploaded file after processing.

Out of scope:

- Background job queueing.
- Partial success for multi-page PDFs.
- New document types beyond PDF and TXT.
- UI changes.

## Current Behavior

The current `/api/upload` implementation stores the uploaded file on disk, then:

- For PDF files, calls the new local OCR path in `DocumentProcessor`.
- For TXT files, reads the file directly.
- Passes the extracted text to `rag.ainsert()`.

This design removes the Qwen VL branch and replaces it with a local OCR extraction service.

## Proposed Architecture

### 1. `DocumentProcessor` service

Create `backend/core/document_processor.py` with a single responsibility: convert an uploaded file path into plain text.

Responsibilities:

- Detect file type from extension.
- For PDF files, convert pages to images using `pdf2image.convert_from_path()`.
- Run PaddleOCR on each page image.
- Concatenate page text in page order.
- For TXT files, read UTF-8 text directly.

### 2. Upload route orchestration

Keep the `/api/upload` endpoint as the orchestration layer:

- Save the uploaded file to a temporary path.
- Call `DocumentProcessor.extract_text()`.
- Validate that extracted text is non-empty.
- Call `rag.ainsert()` with the extracted text.
- Return success only after insert completes.
- Delete the temp file in `finally`.

## Processing Flow

### PDF path

1. Receive PDF upload.
2. Save to temp file.
3. Call `pdf2image.convert_from_path()` inside `asyncio.to_thread()`.
4. For each page image, call PaddleOCR inside `asyncio.to_thread()`.
5. Combine page text into one document string.
6. If any page fails, raise an exception and abort the request.
7. If text is empty, raise an error.
8. Call `rag.ainsert()`.
9. Return success.
10. Delete temp file.

### TXT path

1. Receive TXT upload.
2. Save to temp file.
3. Read file as UTF-8 with `errors="ignore"` or equivalent safe decode behavior.
4. If content is empty, raise an error.
5. Call `rag.ainsert()`.
6. Return success.
7. Delete temp file.

## Error Handling Rules

- Unsupported extension -> `400 Bad Request`.
- PDF conversion failure -> `500 Internal Server Error`.
- Any OCR failure on any page -> `500 Internal Server Error`.
- Empty extracted text -> `500 Internal Server Error`.
- `rag.ainsert()` failure -> `500 Internal Server Error`.
- Temp file deletion should run even after failures.

The request is all-or-nothing. No partial PDF indexing is allowed.

## Document Status

The upload flow relies on LightRAG's existing doc status storage through `rag.ainsert()` and `PGDocStatusStorage`.

Expected behavior:

- `rag.ainsert()` should be the single source of truth for document insertion and status updates.
- The route does not introduce a separate custom status table or duplicate tracking layer.
- Document listing continues to read from `rag.doc_status` as it does today.

## Temp File Cleanup

The uploaded file should be stored in a temporary working path derived from the configured LightRAG working directory.

Cleanup rules:

- Use `try/finally` around the whole upload pipeline.
- Delete the uploaded temp file after processing regardless of success or failure.
- If future PDF-to-image conversion creates additional temp artifacts, delete those too in the same cleanup block.

## Implementation Boundaries

### `backend/core/document_processor.py`

- `DocumentProcessor.extract_text(file_path: str) -> str`
- `DocumentProcessor._extract_pdf(file_path: str) -> str`
- `DocumentProcessor._extract_txt(file_path: str) -> str`

The route should not contain OCR-specific logic.

### `backend/api/routes.py`

- Keep request validation and response formatting here.
- Keep file persistence and cleanup here.
- Call the processor and `rag.ainsert()`.

## Testing Plan

1. Unit test PDF extraction orchestration with mocked `pdf2image` and PaddleOCR.
2. Unit test TXT extraction reads file contents correctly.
3. Unit test any OCR exception aborts the whole request.
4. Integration test `/api/upload` for a TXT file.
5. Integration test `/api/upload` for an unsupported file extension.
6. Verify temp file cleanup occurs on success and on failure.

## Risks

- `pdf2image` depends on Poppler being available in the runtime environment.
- OCR speed may be slower than the previous vision-model path for large PDFs.
- Large PDFs can produce many page images and increase memory use.

These risks are acceptable for the current scope because the goal is a local, deterministic PDF OCR path.

## Success Criteria

- PDF uploads are processed through `pdf2image` + PaddleOCR only.
- TXT uploads still work.
- Any OCR failure stops the entire upload.
- `rag.ainsert()` still receives the extracted text and indexes the document.
- Temporary files are deleted after processing.
