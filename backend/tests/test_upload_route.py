import asyncio

import pytest


class DummyUploadFile:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.file = __import__("io").BytesIO(data)


def test_upload_rejects_unsupported_extension(monkeypatch):
    file = DummyUploadFile("sample.docx", b"fake")

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    with pytest.raises(Exception) as exc_info:
        asyncio.run(routes.upload_file(file))

    assert "Only PDF and TXT files are supported" in str(exc_info.value)


def test_upload_cleans_temp_file_on_failure(tmp_path, monkeypatch):
    from backend.config import settings

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    settings.LIGHTRAG_WORKING_DIR = str(tmp_path)

    file = DummyUploadFile("sample.txt", b"xin chao")

    class FakeRAG:
        async def ainsert(self, *args, **kwargs):
            raise RuntimeError("insert failed")

    class FakeProcessor:
        async def extract_text(self, file_path):
            return "noi dung"

    monkeypatch.setattr(routes, "document_processor", FakeProcessor())
    monkeypatch.setattr(routes, "get_rag_engine", lambda: FakeRAG())

    with pytest.raises(Exception) as exc_info:
        asyncio.run(routes.upload_file(file))

    assert "Failed to index file" in str(exc_info.value)
    assert not (tmp_path / "sample.txt").exists()


def test_upload_splits_chunks_by_paragraphs(tmp_path, monkeypatch):
    from backend.config import settings

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    settings.LIGHTRAG_WORKING_DIR = str(tmp_path)

    file = DummyUploadFile("sample.txt", b"xin chao")

    class FakeRAG:
        def __init__(self):
            self.calls = []

        async def ainsert(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    class FakeProcessor:
        async def extract_text(self, file_path):
            return "doan 1\n\ndoan 2"

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "document_processor", FakeProcessor())
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)

    response = asyncio.run(routes.upload_file(file))

    assert response.status == "success"
    assert fake_rag.calls[0][1]["split_by_character"] == "\n\n"
