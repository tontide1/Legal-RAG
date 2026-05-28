# Docling No-OCR PDF Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the backend PDF ingestion path with Docling configured with OCR disabled, save extracted PDF text artifacts, and remove runtime OCR code and dependencies.

**Architecture:** Keep the existing `/api/upload` and LightRAG insertion contract unchanged. Put Docling behind `DocumentProcessor.extract_text()`, write extracted PDF text to `settings.LIGHTRAG_WORKING_DIR/extracted_txt/{pdf_stem}.txt`, and raise a clear error for PDFs with no embedded text. Remove legacy OCR and vision PDF parsing helpers so backend runtime no longer imports OCR/image-conversion packages.

**Tech Stack:** Python 3.11, FastAPI, LightRAG, Docling `DocumentConverter`, pytest, Docker Compose.

---

## File Structure

- Modify: `backend/core/document_processor.py`
  - Owns PDF/TXT extraction.
  - New PDF path uses Docling with `PdfPipelineOptions.do_ocr = False`.
  - Writes extracted PDF text artifacts under `LIGHTRAG_WORKING_DIR/extracted_txt`.

- Modify: `backend/core/llm_services.py`
  - Remove `qwen_vl_parse_pdf()` because it imports `pdf2image` and `pypdf` for legacy vision PDF parsing.
  - Keep `deepseek_llm_func = gemini_chat_llm_func`.

- Modify: `backend/tests/test_document_processor.py`
  - Replace OCR page test with Docling no-OCR extraction tests.
  - Keep TXT extraction test.

- Modify: `backend/requirements.txt`
  - Add `docling`.
  - Remove `pymupdf4llm`, `pypdf`, `pdf2image`, `paddleocr`, and old OCR comments.

- Modify: `backend/Dockerfile`
  - Remove `poppler-utils` from apt packages.
  - Remove `pip install paddlepaddle==3.2.1 ...`.

- Modify: `README.md`, `ARCHITECTURE.md`, `PROJECT_STRUCTURE.md`
  - Replace old OCR/PaddleOCR/Chandra/Qwen-VL PDF parsing descriptions with Docling no-OCR behavior.

- Update Serena memories:
  - `mem:backend/core`
  - `mem:tech_stack`
  - `mem:conventions`

---

### Task 1: Add Failing Docling Extraction Tests

**Files:**
- Modify: `backend/tests/test_document_processor.py`

- [ ] **Step 1: Replace the current file with tests for TXT, Docling no-OCR PDF extraction, artifact writing, and empty PDF failure**

```python
import asyncio
import sys
import types

import pytest


def _install_fake_docling(monkeypatch, exported_text: str, calls: dict):
    docling_module = types.ModuleType("docling")
    document_converter_module = types.ModuleType("docling.document_converter")
    datamodel_module = types.ModuleType("docling.datamodel")
    base_models_module = types.ModuleType("docling.datamodel.base_models")
    pipeline_options_module = types.ModuleType("docling.datamodel.pipeline_options")

    class FakeInputFormat:
        PDF = "pdf"

    class FakePdfPipelineOptions:
        def __init__(self):
            self.do_ocr = True

    class FakePdfFormatOption:
        def __init__(self, pipeline_options):
            self.pipeline_options = pipeline_options
            calls["pdf_pipeline_options"] = pipeline_options

    class FakeDocument:
        def export_to_markdown(self):
            calls["export_called"] = True
            return exported_text

    class FakeDocumentConverter:
        def __init__(self, format_options):
            calls["format_options"] = format_options

        def convert(self, source):
            calls["source"] = source
            return types.SimpleNamespace(document=FakeDocument())

    document_converter_module.DocumentConverter = FakeDocumentConverter
    document_converter_module.PdfFormatOption = FakePdfFormatOption
    base_models_module.InputFormat = FakeInputFormat
    pipeline_options_module.PdfPipelineOptions = FakePdfPipelineOptions

    monkeypatch.setitem(sys.modules, "docling", docling_module)
    monkeypatch.setitem(sys.modules, "docling.document_converter", document_converter_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel", datamodel_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", base_models_module)
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", pipeline_options_module)


def test_extract_text_reads_txt_file(tmp_path):
    from backend.core.document_processor import DocumentProcessor

    file_path = tmp_path / "sample.txt"
    file_path.write_text("Xin chao legal RAG", encoding="utf-8")

    processor = DocumentProcessor()

    text = asyncio.run(processor.extract_text(str(file_path)))

    assert text == "Xin chao legal RAG"


def test_extract_pdf_uses_docling_without_ocr_and_writes_txt(tmp_path, monkeypatch):
    from backend.config import settings
    from backend.core.document_processor import DocumentProcessor

    monkeypatch.setattr(settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%fake pdf file\n")

    calls = {}
    _install_fake_docling(monkeypatch, "Dieu 1. Noi dung van ban", calls)

    text = asyncio.run(DocumentProcessor().extract_text(str(file_path)))

    assert text == "Dieu 1. Noi dung van ban"
    assert calls["source"] == str(file_path)
    assert calls["pdf_pipeline_options"].do_ocr is False
    assert calls["export_called"] is True
    assert (tmp_path / "extracted_txt" / "sample.txt").read_text(encoding="utf-8") == text


def test_extract_pdf_raises_when_docling_returns_empty_text(tmp_path, monkeypatch):
    from backend.config import settings
    from backend.core.document_processor import DocumentProcessor

    monkeypatch.setattr(settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%fake scan pdf file\n")

    calls = {}
    _install_fake_docling(monkeypatch, "   ", calls)

    with pytest.raises(ValueError, match="OCR is disabled, so scanned PDFs are not supported"):
        asyncio.run(DocumentProcessor().extract_text(str(file_path)))

    assert calls["pdf_pipeline_options"].do_ocr is False
    assert not (tmp_path / "extracted_txt" / "scan.txt").exists()
```

- [ ] **Step 2: Run the focused test file to verify it fails before implementation**

Run:

```bash
pytest backend/tests/test_document_processor.py -v
```

Expected: the TXT test passes; the two PDF tests fail because `DocumentProcessor` still uses old PDF/OCR helpers and does not create a stem-derived `.txt` artifact under `extracted_txt`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add backend/tests/test_document_processor.py
git commit -m "test: specify docling no-ocr pdf extraction"
```

---

### Task 2: Implement Docling No-OCR PDF Extraction

**Files:**
- Modify: `backend/core/document_processor.py`
- Test: `backend/tests/test_document_processor.py`

- [ ] **Step 1: Replace `backend/core/document_processor.py` with the Docling implementation**

```python
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
    """Async document processor supporting text-layer PDF and TXT files."""

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
```

- [ ] **Step 2: Run the focused document processor tests**

Run:

```bash
pytest backend/tests/test_document_processor.py -v
```

Expected: all tests in `test_document_processor.py` pass.

- [ ] **Step 3: Run upload route tests to verify the `extract_text()` contract still works**

Run:

```bash
pytest backend/tests/test_upload_route.py -v
```

Expected: all upload route tests pass without route changes.

- [ ] **Step 4: Commit the Docling processor implementation**

```bash
git add backend/core/document_processor.py backend/tests/test_document_processor.py
git commit -m "feat: extract pdf text with docling without ocr"
```

---

### Task 3: Remove Legacy Vision PDF Helper From LLM Services

**Files:**
- Modify: `backend/core/llm_services.py`

- [ ] **Step 1: Verify `qwen_vl_parse_pdf` has no callers**

Run:

```bash
rg -n "qwen_vl_parse_pdf" backend
```

Expected before deletion:

One match in `backend/core/llm_services.py` defining `async def qwen_vl_parse_pdf(file_path: str) -> str:`.

- [ ] **Step 2: Delete `qwen_vl_parse_pdf()` from `backend/core/llm_services.py`**

After deletion, the end of the file should keep this alias and no PDF parsing helper after it:

```python
# Backward-compatible alias for existing imports/tests.
deepseek_llm_func = gemini_chat_llm_func
```

- [ ] **Step 3: Verify the helper is gone**

Run:

```bash
rg -n "qwen_vl_parse_pdf|pdf2image|pypdf" backend/core/llm_services.py
```

Expected: no output.

- [ ] **Step 4: Run the existing RAG engine tests**

Run:

```bash
pytest backend/tests/test_rag_engine.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the legacy helper removal**

```bash
git add backend/core/llm_services.py
git commit -m "refactor: remove legacy vision pdf parser"
```

---

### Task 4: Update Backend Dependencies and Dockerfile

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/Dockerfile`

- [ ] **Step 1: Replace the document processing block in `backend/requirements.txt`**

Use this block:

```text
# Document Processing
docling>=2.81.0
python-docx>=1.1.0
openpyxl>=3.1.0
python-pptx>=0.6.0
```

The file must no longer include these lines:

```text
pymupdf4llm>=0.0.17
pypdf>=4.0.0
pdf2image>=1.17.0
paddleocr
```

- [ ] **Step 2: Remove Poppler and PaddlePaddle install from `backend/Dockerfile`**

The apt package block should become:

```dockerfile
RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  gcc \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*
```

Remove this Dockerfile block entirely:

```dockerfile
# Install paddlepaddle from PaddlePaddle's official CPU index (not on PyPI)
RUN pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

- [ ] **Step 3: Verify removed runtime dependencies are gone from backend config files**

Run:

```bash
rg -n "pymupdf4llm|pypdf|pdf2image|paddleocr|paddlepaddle|poppler-utils" backend/requirements.txt backend/Dockerfile
```

Expected: no output.

- [ ] **Step 4: Commit dependency and Dockerfile cleanup**

```bash
git add backend/requirements.txt backend/Dockerfile
git commit -m "chore: replace pdf ocr dependencies with docling"
```

---

### Task 5: Update User-Facing Documentation

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PROJECT_STRUCTURE.md`

- [ ] **Step 1: Update README feature and tech stack text**

In `README.md`, replace the old PDF/OCR feature bullet with:

```markdown
- **Docling PDF Parsing Without OCR**: Uses **Docling** to extract embedded text from legal PDFs and saves extracted `.txt` artifacts for indexing. Scanned/image-only PDFs are rejected because OCR is disabled by design.
```

In `README.md`, replace the LLM/Embeddings tech stack bullet with:

```markdown
- **LLM/Embeddings**: Gemini Developer API for chat generation, Ollama for LightRAG indexing extraction, local Vietnamese legal embeddings with `huyydangg/DEk21_hcmute_embedding`, and Docling for no-OCR PDF text extraction.
```

- [ ] **Step 2: Update `ARCHITECTURE.md` component and flow descriptions**

Replace the `DocumentProcessor` row with:

```markdown
| **DocumentProcessor** | Converts PDF or plain TXT to text. Uses Docling for PDFs with OCR disabled, writes extracted PDF text to `extracted_txt/{stem}.txt`, and rejects scanned/image-only PDFs with no embedded text. | `backend/core/document_processor.py` |
```

Replace this ingestion bullet:

```markdown
- `DocumentProcessor.extract_text` -> extracts text directly or via OCR.
```

with:

```markdown
- `DocumentProcessor.extract_text` -> extracts embedded PDF text through Docling with OCR disabled, or reads TXT directly.
```

Replace the pre-processing pipeline file type line with:

```markdown
- **File type detection** -> TXT files are read directly; PDFs are converted through Docling with OCR disabled. Scanned/image-only PDFs fail clearly instead of entering an OCR fallback.
```

- [ ] **Step 3: Update `PROJECT_STRUCTURE.md` backend tree comment**

Replace the `document_processor.py` comment with:

```markdown
│   │   ├─ document_processor.py  # Extracts text from PDF/TXT via Docling without OCR
```

- [ ] **Step 4: Verify old OCR descriptions are gone from user-facing docs**

Run:

```bash
rg -n "PaddleOCR|Chandra OCR|Vision-Based PDF Parsing|Qwen 3 VL|pymupdf4llm|pdf2image" README.md ARCHITECTURE.md PROJECT_STRUCTURE.md
```

Expected: no output.

- [ ] **Step 5: Commit documentation updates**

```bash
git add README.md ARCHITECTURE.md PROJECT_STRUCTURE.md
git commit -m "docs: describe docling pdf extraction without ocr"
```

---

### Task 6: Update Serena Memories

**Files:**
- Serena memory: `mem:backend/core`
- Serena memory: `mem:tech_stack`
- Serena memory: `mem:conventions`

- [ ] **Step 1: Read current memories**

Use Serena:

```text
read_memory("mem:backend/core")
read_memory("mem:tech_stack")
read_memory("mem:conventions")
```

Expected: current memories still mention OCR, PaddleOCR, Chandra, or old PDF extraction.

- [ ] **Step 2: Update `mem:backend/core` ingestion notes**

Replace the ingestion pipeline notes with these stable facts:

```markdown
Ingestion pipeline:
- `DocumentProcessor.extract_text()` dispatches by suffix.
- PDF path uses Docling with OCR disabled and writes extracted text to `settings.LIGHTRAG_WORKING_DIR/extracted_txt/{pdf_stem}.txt`.
- PDFs with no embedded/extractable text fail clearly; scanned/image-only PDFs are not supported.
- TXT path reads UTF-8 directly.
- `normalize_for_lightrag()` is the canonical ingest normalization path used before `rag.ainsert()`.
- `chunk_markdown()` exists for standalone/offline use; the main upload path uses LightRAG splitting on `\n\n` after normalization.
```

- [ ] **Step 3: Update `mem:tech_stack` backend notes**

Replace the PDF/OCR extraction bullets with:

```markdown
Backend:
- Python 3.11
- FastAPI
- LightRAG
- PostgreSQL with pgvector and Apache AGE
- PDF text extraction: Docling with OCR disabled
- Scanned/image-only PDFs are intentionally rejected because there is no OCR step
```

- [ ] **Step 4: Update `mem:conventions` backend notes**

Add this convention under backend:

```markdown
- Keep PDF ingestion no-OCR: do not add PaddleOCR, Chandra OCR, image conversion, or vision-model parsing back into the upload path.
- PDF extraction artifacts live under `settings.LIGHTRAG_WORKING_DIR/extracted_txt`.
```

- [ ] **Step 5: Run memory sanity check**

Run:

```bash
serena memories check
```

Expected: no broken memory references.

- [ ] **Step 6: Commit memory updates if Serena writes tracked files**

Run:

```bash
git status --short .serena
```

If `.serena` files are tracked or intentionally part of this repo, commit them:

```bash
git add .serena
git commit -m "docs: update serena no-ocr ingestion notes"
```

If `.serena` remains untracked local agent state, leave it unstaged and do not commit it.

---

### Task 7: Run Full Verification and Final Cleanup

**Files:**
- Verify repository state only.

- [ ] **Step 1: Run backend test suite**

Run:

```bash
pytest backend/tests
```

Expected: all backend tests pass.

- [ ] **Step 2: Verify removed runtime OCR and legacy PDF packages are not referenced by backend or user-facing docs**

Run:

```bash
rg -n "paddleocr|paddlepaddle|pdf2image|pymupdf4llm|pypdf|chandra_ocr|qwen_vl_parse_pdf|PaddleOCR|Chandra OCR|Vision-Based PDF Parsing|Qwen 3 VL" backend README.md ARCHITECTURE.md PROJECT_STRUCTURE.md
```

Expected: no output.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional changes are present. No generated data, `.env`, model files, raw data, or extracted PDF artifacts are staged.

- [ ] **Step 4: Commit remaining intentional changes if any task left them uncommitted**

Run:

```bash
git add backend/core/document_processor.py backend/core/llm_services.py backend/tests/test_document_processor.py backend/requirements.txt backend/Dockerfile README.md ARCHITECTURE.md PROJECT_STRUCTURE.md
git commit -m "feat: use docling pdf extraction without ocr"
```

Expected: either a commit is created for remaining changes, or git reports nothing to commit because prior task commits already captured every intentional change.
