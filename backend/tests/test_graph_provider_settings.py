import asyncio
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


GRAPH_PROVIDER_KEY = "graph_build_provider"


class FakeConnection:
    def __init__(self, initial_value=None):
        self.values = (
            {GRAPH_PROVIDER_KEY: initial_value} if initial_value is not None else {}
        )
        self.execute_calls = []
        self.fetchval_calls = []

    async def execute(self, sql, *args):
        self.execute_calls.append((sql, args))
        statement = sql.strip().upper()
        if statement.startswith("INSERT INTO APP_SETTINGS") or statement.startswith(
            "UPDATE APP_SETTINGS"
        ):
            key, value = args
            self.values[key] = value
        return "OK"

    async def fetchval(self, sql, *args):
        self.fetchval_calls.append((sql, args))
        return self.values.get(args[0])


class FakeAcquireContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False

    def acquire(self):
        return FakeAcquireContext(self.connection)

    async def close(self):
        self.closed = True


@pytest.fixture
def app_settings_module(monkeypatch):
    import backend.core.app_settings as app_settings

    monkeypatch.setattr(
        app_settings,
        "settings",
        types.SimpleNamespace(
            DATABASE_URL="postgresql://user:pass@localhost:5432/db",
            GRAPH_BUILD_PROVIDER_DEFAULT="ollama",
            DEFAULT_GRAPH_BUILD_PROVIDER="ollama",
            NINE_ROUTER_BASE_URL="https://router.example",
            NINE_ROUTER_HEALTH_PATH="/health",
            NINE_ROUTER_API_KEY="api-key",
            NINE_ROUTER_TIMEOUT_SECONDS=2,
        ),
    )
    return app_settings


def test_get_graph_build_provider_bootstraps_default_when_missing(
    app_settings_module, monkeypatch
):
    connection = FakeConnection()
    pool = FakePool(connection)

    async def fake_create_pool(**kwargs):
        return pool

    monkeypatch.setattr(app_settings_module.asyncpg, "create_pool", fake_create_pool)

    service = app_settings_module.AppSettingsService()

    asyncio.run(service.initialize())
    assert connection.values[GRAPH_PROVIDER_KEY] == "ollama"

    provider = asyncio.run(service.get_graph_build_provider())

    assert provider == "ollama"
    assert "CREATE TABLE IF NOT EXISTS app_settings" in connection.execute_calls[0][0]


def test_set_graph_build_provider_rejects_unsupported_values(
    app_settings_module, monkeypatch
):
    connection = FakeConnection("ollama")
    pool = FakePool(connection)

    async def fake_create_pool(**kwargs):
        return pool

    monkeypatch.setattr(app_settings_module.asyncpg, "create_pool", fake_create_pool)

    service = app_settings_module.AppSettingsService()

    with pytest.raises(ValueError, match="Unsupported graph_build_provider"):
        asyncio.run(service.set_graph_build_provider("bad-provider"))

    assert connection.values[GRAPH_PROVIDER_KEY] == "ollama"
    assert connection.execute_calls == []


def test_set_graph_build_provider_validates_9router_before_persisting(
    app_settings_module, monkeypatch
):
    connection = FakeConnection("ollama")
    pool = FakePool(connection)
    events = []

    async def fake_create_pool(**kwargs):
        return pool

    async def fake_validate():
        events.append("validated")

    monkeypatch.setattr(app_settings_module.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(
        app_settings_module, "validate_9router_connection", fake_validate
    )

    service = app_settings_module.AppSettingsService()

    asyncio.run(service.set_graph_build_provider("9router"))

    assert events == ["validated"]
    assert connection.values[GRAPH_PROVIDER_KEY] == "9router"
    assert any("INSERT INTO app_settings" in sql for sql, _ in connection.execute_calls)


def test_set_graph_build_provider_keeps_existing_value_when_9router_validation_fails(
    app_settings_module, monkeypatch
):
    connection = FakeConnection("ollama")
    pool = FakePool(connection)

    async def fake_create_pool(**kwargs):
        return pool

    async def fake_validate():
        raise RuntimeError("router unreachable")

    monkeypatch.setattr(app_settings_module.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(
        app_settings_module, "validate_9router_connection", fake_validate
    )

    service = app_settings_module.AppSettingsService()

    with pytest.raises(RuntimeError, match="router unreachable"):
        asyncio.run(service.set_graph_build_provider("9router"))

    assert connection.values[GRAPH_PROVIDER_KEY] == "ollama"
    assert connection.execute_calls == []


def test_validate_9router_connection_accepts_reasoning_only_response(
    app_settings_module, monkeypatch
):
    class FakeCompletions:
        async def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="",
                            model_extra={"reasoning_content": "OK"},
                        )
                    )
                ]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(
        app_settings_module, "_get_nine_router_client", lambda: FakeClient()
    )
    monkeypatch.setattr(
        app_settings_module.settings,
        "NINE_ROUTER_INDEX_MODEL",
        "openrouter/gpt-oss-20b",
        raising=False,
    )

    asyncio.run(app_settings_module.validate_9router_connection())


def test_lifespan_initializes_and_closes_settings_service(monkeypatch):
    events = []

    class FakeRAGEngine:
        @classmethod
        async def initialize(cls):
            events.append("rag.initialize")

        @classmethod
        async def finalize(cls):
            events.append("rag.finalize")

    lightrag_module = types.ModuleType("lightrag")
    lightrag_module.__path__ = []
    lightrag_module.LightRAG = object

    rerank_module = types.ModuleType("lightrag.rerank")
    rerank_module.jina_rerank = lambda *args, **kwargs: None

    utils_module = types.ModuleType("lightrag.utils")
    utils_module.EmbeddingFunc = object

    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    monkeypatch.setitem(sys.modules, "lightrag.rerank", rerank_module)
    monkeypatch.setitem(sys.modules, "lightrag.utils", utils_module)
    sys.modules.pop("backend.main", None)
    sys.modules.pop("backend.core.rag_engine", None)

    try:
        import backend.main as main

        async def fake_initialize_graph_provider_settings():
            events.append("settings.initialize")

        async def fake_close_graph_provider_settings():
            events.append("settings.close")

        monkeypatch.setattr(
            main,
            "initialize_graph_provider_settings",
            fake_initialize_graph_provider_settings,
        )
        monkeypatch.setattr(
            main, "close_graph_provider_settings", fake_close_graph_provider_settings
        )
        monkeypatch.setattr(main, "RAGEngine", FakeRAGEngine)

        async def run_lifespan():
            async with main.lifespan(types.SimpleNamespace()):
                events.append("inside")

        asyncio.run(run_lifespan())

        assert events == [
            "settings.initialize",
            "rag.initialize",
            "inside",
            "rag.finalize",
            "settings.close",
        ]

    finally:
        sys.modules.pop("backend.main", None)
        sys.modules.pop("backend.core.rag_engine", None)
