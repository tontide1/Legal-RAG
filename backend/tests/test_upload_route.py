import asyncio

import pytest


class DummyUploadFile:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.file = __import__("io").BytesIO(data)


class FakeDocStatus:
    async def get_docs_paginated(self):
        return ([], None)


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
        def __init__(self):
            self.doc_status = FakeDocStatus()

        async def ainsert(self, *args, **kwargs):
            raise RuntimeError("insert failed")

    class FakeProcessor:
        async def extract_text(self, file_path):
            return "noi dung"

    monkeypatch.setattr(routes, "document_processor", FakeProcessor())
    
    async def fake_get_ingest_rag_engine(provider: str = "ollama"):
        return FakeRAG()

    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(routes.upload_file(file))

    assert "Failed to index file" in str(exc_info.value)
    assert not (tmp_path / "sample.txt").exists()


def test_upload_normalizes_and_uses_lightrag_split(tmp_path, monkeypatch):
    from backend.config import settings

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    settings.LIGHTRAG_WORKING_DIR = str(tmp_path)

    file = DummyUploadFile("sample.txt", b"xin chao")

    class FakeRAG:
        def __init__(self):
            self.calls = []
            self.doc_status = FakeDocStatus()

        async def ainsert(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    class FakeProcessor:
        async def extract_text(self, file_path):
            return "doan 1\n\ndoan 2"

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "document_processor", FakeProcessor())
    
    async def fake_get_ingest_rag_engine(provider: str = "ollama"):
        return fake_rag

    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)

    response = asyncio.run(routes.upload_file(file))

    assert response.status == "success"
    assert fake_rag.calls[0][0][0] == "doan 1\n\ndoan 2"
    assert fake_rag.calls[0][1]["split_by_character"] == "\n\n"
    assert fake_rag.calls[0][1]["file_paths"] == ["sample.txt"]


def test_upload_returns_conflict_for_existing_document(tmp_path, monkeypatch):
    from backend.config import settings

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    settings.LIGHTRAG_WORKING_DIR = str(tmp_path)

    file = DummyUploadFile("sample.txt", b"xin chao")

    class StatusValue:
        def __init__(self, value):
            self.value = value

    class ExistingDocStatus:
        async def get_docs_paginated(self):
            status_obj = type(
                "StatusObj",
                (),
                {
                    "file_path": "sample.txt",
                    "status": StatusValue("processed"),
                },
            )()
            return ((("doc-1", status_obj),), None)

    class FakeRAG:
        def __init__(self):
            self.doc_status = ExistingDocStatus()

        async def ainsert(self, *args, **kwargs):
            raise AssertionError("ainsert should not be called for duplicate files")

    class FakeProcessor:
        async def extract_text(self, file_path):
            raise AssertionError("extract_text should not run for duplicate files")

    monkeypatch.setattr(routes, "document_processor", FakeProcessor())
    
    async def fake_get_ingest_rag_engine(provider: str = "ollama"):
        return FakeRAG()

    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(routes.upload_file(file))

    assert "409" in str(exc_info.value)
