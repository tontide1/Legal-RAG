import asyncio
import sys
import types

import pytest


def _install_fake_docling(monkeypatch, exported_text: str):
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

    class FakeDocument:
        def __init__(self, text):
            self._text = text

        def export_to_markdown(self):
            return self._text

    class FakeDocumentConverter:
        def __init__(self, format_options):
            self.format_options = format_options

        def convert(self, source):
            pdf_options = self.format_options[FakeInputFormat.PDF].pipeline_options
            text = exported_text if pdf_options.do_ocr is False else "OCR SHOULD BE DISABLED"
            return types.SimpleNamespace(document=FakeDocument(text))

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

    _install_fake_docling(monkeypatch, "Dieu 1. Noi dung van ban")

    text = asyncio.run(DocumentProcessor().extract_text(str(file_path)))

    assert text == "Dieu 1. Noi dung van ban"
    assert (tmp_path / "extracted_txt" / "sample.txt").read_text(encoding="utf-8") == text


def test_extract_pdf_raises_when_docling_returns_empty_text(tmp_path, monkeypatch):
    from backend.config import settings
    from backend.core.document_processor import DocumentProcessor

    monkeypatch.setattr(settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%fake scan pdf file\n")

    _install_fake_docling(monkeypatch, "   ")

    with pytest.raises(ValueError):
        asyncio.run(DocumentProcessor().extract_text(str(file_path)))

    assert not (tmp_path / "extracted_txt" / "scan.txt").exists()


def test_extract_pdf_raises_when_docling_returns_image_placeholder(tmp_path, monkeypatch):
    from backend.config import settings
    from backend.core.document_processor import DocumentProcessor

    monkeypatch.setattr(settings, "LIGHTRAG_WORKING_DIR", str(tmp_path))
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%fake scan pdf file\n")

    _install_fake_docling(monkeypatch, "<!-- image -->")

    with pytest.raises(ValueError):
        asyncio.run(DocumentProcessor().extract_text(str(file_path)))

    assert not (tmp_path / "extracted_txt" / "scan.txt").exists()
