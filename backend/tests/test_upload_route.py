import asyncio
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class DummyUploadFile:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.file = __import__("io").BytesIO(data)


class FakeDocStatus:
    async def get_docs_paginated(self):
        return ([], None)


class FakeGraphProviderSettingsService:
    def __init__(self, provider: str = "ollama"):
        self.provider = provider
        self.get_calls = 0
        self.set_calls = []

    async def get_graph_build_provider(self):
        self.get_calls += 1
        return self.provider

    async def set_graph_build_provider(self, provider: str):
        self.set_calls.append(provider)
        self.provider = provider.strip().lower()
        return self.provider


def test_graph_provider_settings_get_returns_current_provider(monkeypatch):
    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    service = FakeGraphProviderSettingsService("9router")
    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: service)

    response = asyncio.run(routes.get_graph_provider_settings())

    assert response.provider == "9router"
    assert service.get_calls == 1


def test_graph_provider_settings_options_returns_allowed_providers(monkeypatch):
    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    response = asyncio.run(routes.get_graph_provider_options())

    assert [option.value for option in response.options] == ["ollama", "9router"]
    assert [option.label for option in response.options] == ["Ollama", "9router Local"]


def test_graph_provider_settings_put_persists_provider_and_returns_status(monkeypatch):
    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    service = FakeGraphProviderSettingsService("ollama")
    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: service)

    response = asyncio.run(
        routes.update_graph_provider_settings(
            routes.GraphProviderSettingsRequest(provider=" 9router ")
        )
    )

    assert response.provider == "9router"
    assert response.status == "success"
    assert "9router" in response.message
    assert service.set_calls == [" 9router "]


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
    scheduled_coroutines = []

    class FakeRAG:
        def __init__(self):
            self.doc_status = FakeDocStatus()

        async def ainsert(self, *args, **kwargs):
            raise RuntimeError("insert failed")

    class FakeProcessor:
        async def extract_text(self, file_path):
            return "noi dung"

    monkeypatch.setattr(routes, "document_processor", FakeProcessor())

    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: FakeGraphProviderSettingsService("ollama"))

    async def fake_get_ingest_rag_engine(provider: str = "ollama"):
        assert provider == "ollama"
        return FakeRAG()

    def fake_create_task(coro):
        scheduled_coroutines.append(coro)

        class DummyTask:
            def add_done_callback(self, callback):
                self.callback = callback

        return DummyTask()

    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)
    monkeypatch.setattr(routes.asyncio, "create_task", fake_create_task)

    response = asyncio.run(routes.upload_file(file))

    assert response.status == "success"
    assert len(scheduled_coroutines) == 1
    assert not (tmp_path / "sample.txt").exists()

    with pytest.raises(RuntimeError, match="insert failed"):
        asyncio.run(scheduled_coroutines[0])


def test_upload_normalizes_and_uses_lightrag_split(tmp_path, monkeypatch):
    from backend.config import settings

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    settings.LIGHTRAG_WORKING_DIR = str(tmp_path)

    file = DummyUploadFile("sample.txt", b"xin chao")
    scheduled_coroutines = []

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

    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: FakeGraphProviderSettingsService("9router"))

    async def fake_get_ingest_rag_engine(provider: str = "ollama"):
        assert provider == "9router"
        return fake_rag

    def fake_create_task(coro):
        scheduled_coroutines.append(coro)

        class DummyTask:
            def add_done_callback(self, callback):
                self.callback = callback

        return DummyTask()

    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)
    monkeypatch.setattr(routes.asyncio, "create_task", fake_create_task)

    response = asyncio.run(routes.upload_file(file))

    assert response.status == "success"
    assert "9router" in response.message
    assert len(scheduled_coroutines) == 1

    asyncio.run(scheduled_coroutines[0])

    assert fake_rag.calls[0][0][0] == "doan 1\n\ndoan 2"
    assert fake_rag.calls[0][1]["split_by_character"] == "\n\n"
    assert fake_rag.calls[0][1]["file_paths"] == ["sample.txt"]


def test_upload_schedules_background_indexing_for_accepted_file(tmp_path, monkeypatch):
    from backend.config import settings

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    settings.LIGHTRAG_WORKING_DIR = str(tmp_path)

    file = DummyUploadFile("sample.txt", b"xin chao")
    events = []

    class FakeRAG:
        def __init__(self):
            self.doc_status = FakeDocStatus()

        async def ainsert(self, *args, **kwargs):
            events.append(("ainsert", args, kwargs))

    class FakeProcessor:
        async def extract_text(self, file_path):
            events.append(("extract_text", file_path))
            return "doan 1\n\ndoan 2"

    fake_rag = FakeRAG()
    scheduled_coroutines = []

    monkeypatch.setattr(routes, "document_processor", FakeProcessor())
    monkeypatch.setattr(
        routes,
        "get_graph_provider_settings_service",
        lambda: FakeGraphProviderSettingsService("9router"),
    )

    async def fake_get_ingest_rag_engine(provider: str = "ollama"):
        assert provider == "9router"
        return fake_rag

    def fake_create_task(coro):
        scheduled_coroutines.append(coro)

        class DummyTask:
            def add_done_callback(self, callback):
                self.callback = callback

        return DummyTask()

    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)
    monkeypatch.setattr(routes.asyncio, "create_task", fake_create_task)

    response = asyncio.run(routes.upload_file(file))

    assert response.status == "success"
    assert "processing in background" in response.message
    assert len(scheduled_coroutines) == 1
    assert [event[0] for event in events] == ["extract_text"]

    asyncio.run(scheduled_coroutines[0])

    assert [event[0] for event in events] == ["extract_text", "ainsert"]


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

    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: FakeGraphProviderSettingsService("ollama"))

    async def fake_get_ingest_rag_engine(provider: str = "ollama"):
        assert provider == "ollama"
        return FakeRAG()

    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(routes.upload_file(file))

    assert "409" in str(exc_info.value)
