import asyncio


def test_extract_text_reads_txt_file(tmp_path):
    from backend.core.document_processor import DocumentProcessor

    file_path = tmp_path / "sample.txt"
    file_path.write_text("Xin chao legal RAG", encoding="utf-8")

    processor = DocumentProcessor()

    text = asyncio.run(processor.extract_text(str(file_path)))

    assert text == "Xin chao legal RAG"


def test_extract_text_ocrs_each_pdf_page(tmp_path, monkeypatch):
    from backend.core.document_processor import DocumentProcessor

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n%fake pdf file\n")

    page_one = object()
    page_two = object()

    def fake_convert_from_path(path):
        assert path == str(file_path)
        return [page_one, page_two]

    class FakeOCR:
        def __init__(self):
            self.calls = []

        def ocr(self, image, cls=True):
            self.calls.append((image, cls))
            if image is page_one:
                return [[[[0, 0], [1, 1]], ("Trang 1", 0.99)]]
            return [[[[0, 0], [1, 1]], ("Trang 2", 0.98)]]

    fake_ocr = FakeOCR()

    monkeypatch.setattr("backend.core.document_processor._convert_pdf_to_images", fake_convert_from_path)

    processor = DocumentProcessor(ocr_engine=fake_ocr)

    text = asyncio.run(processor.extract_text(str(file_path)))

    assert text == "Trang 1\n\nTrang 2"
    assert fake_ocr.calls == [(page_one, True), (page_two, True)]
