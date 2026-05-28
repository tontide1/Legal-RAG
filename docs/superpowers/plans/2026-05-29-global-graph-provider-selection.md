# Global Graph Provider Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a global UI-controlled setting that switches graph-building/indexing between `ollama` and `9router local`, persists that choice in PostgreSQL, and validates 9router before saving it.

**Architecture:** Add a small async PostgreSQL-backed settings service for `graph_build_provider`, bootstrap it at backend startup, and expose `GET/PUT` settings endpoints. Keep query/chat behavior unchanged, refactor ingest selection in `RAGEngine` to be provider-aware at runtime, add a new OpenAI-compatible 9router indexing wrapper, and update the upload UI to edit the global setting before future uploads.

**Tech Stack:** Python 3.11, FastAPI, asyncpg, LightRAG, OpenAI Python SDK, Ollama Python SDK, React 18, TypeScript, Vite, pytest.

---

## File Structure

- Create: `backend/core/app_settings.py`
  - Own the `app_settings` table bootstrap, asyncpg pool lifecycle, provider read/write methods, and 9router-save validation hook.

- Modify: `backend/config.py`
  - Add default graph-provider setting and 9router local configuration.

- Modify: `backend/main.py`
  - Initialize and close the app-settings persistence layer alongside the existing `RAGEngine` lifespan.

- Modify: `backend/core/llm_services.py`
  - Add a cached OpenAI-compatible 9router client, a 9router indexing wrapper, and a lightweight 9router validation request.

- Modify: `backend/core/rag_engine.py`
  - Replace single ingest singleton with provider-keyed ingest cache and runtime selection.

- Modify: `backend/api/schemas.py`
  - Add request/response models for graph-provider settings endpoints.

- Modify: `backend/api/routes.py`
  - Add settings endpoints, make upload read the persisted provider, and include the chosen provider in upload success messages.

- Create: `backend/tests/test_graph_provider_settings.py`
  - Cover settings bootstrap, valid/invalid provider writes, and failed 9router validation behavior.

- Modify: `backend/tests/test_rag_engine.py`
  - Cover provider-aware ingest instance selection and 9router config wiring.

- Modify: `backend/tests/test_upload_route.py`
  - Cover settings endpoints and upload/provider routing behavior.

- Modify: `frontend/src/components/FileUpload.tsx`
  - Add global provider selector, load/save state, 9router validation feedback, and provider-aware upload messaging.

- Modify: `.env.example`, `README.md`, `ARCHITECTURE.md`, `PROJECT_STRUCTURE.md`
  - Document new environment variables, backend/provider architecture, and UI behavior.

---

### Task 1: Add Persistent Graph-Provider Settings Backend

**Files:**
- Create: `backend/tests/test_graph_provider_settings.py`
- Create: `backend/core/app_settings.py`
- Modify: `backend/config.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write the failing settings-service tests**

Create `backend/tests/test_graph_provider_settings.py`:

```python
import asyncio
import types

import pytest


class FakeConnection:
    def __init__(self, state):
        self.state = state
        self.closed = False

    async def execute(self, sql, *args):
        compact_sql = " ".join(sql.split()).lower()

        if "create table if not exists app_settings" in compact_sql:
            self.state["table_created"] = True
            return "CREATE TABLE"

        if compact_sql.startswith("insert into app_settings"):
            key, value = args
            self.state.setdefault("rows", {}).setdefault(
                key,
                {"key": key, "value": value, "updated_at": "now"},
            )
            return "INSERT 0 1"

        if compact_sql.startswith("update app_settings"):
            value, key = args
            self.state.setdefault("rows", {})[key] = {
                "key": key,
                "value": value,
                "updated_at": "now",
            }
            return "UPDATE 1"

        raise AssertionError(f"Unexpected SQL: {sql}")

    async def fetchrow(self, sql, *args):
        compact_sql = " ".join(sql.split()).lower()

        if "select value from app_settings" not in compact_sql:
            raise AssertionError(f"Unexpected SQL: {sql}")

        key = args[0]
        row = self.state.setdefault("rows", {}).get(key)
        if row is None:
            return None
        return {"value": row["value"]}


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, state):
        self.state = state
        self.connection = FakeConnection(state)

    def acquire(self):
        return FakeAcquire(self.connection)

    async def close(self):
        self.state["pool_closed"] = True


def test_get_graph_build_provider_bootstraps_default():
    from backend.core.app_settings import GraphProviderSettingsService

    state = {}
    service = GraphProviderSettingsService(
        pool=FakePool(state),
        default_provider="ollama",
        nine_router_validator=lambda: asyncio.sleep(0),
    )

    provider = asyncio.run(service.get_graph_build_provider())

    assert provider == "ollama"
    assert state["table_created"] is True
    assert state["rows"]["graph_build_provider"]["value"] == "ollama"


def test_set_graph_build_provider_rejects_unsupported_value():
    from backend.core.app_settings import GraphProviderSettingsService

    state = {}
    service = GraphProviderSettingsService(
        pool=FakePool(state),
        default_provider="ollama",
        nine_router_validator=lambda: asyncio.sleep(0),
    )

    with pytest.raises(ValueError, match="Unsupported graph build provider"):
        asyncio.run(service.set_graph_build_provider("gemini"))


def test_set_graph_build_provider_validates_nine_router_before_save():
    from backend.core.app_settings import GraphProviderSettingsService

    state = {"rows": {"graph_build_provider": {"key": "graph_build_provider", "value": "ollama"}}}
    calls = []

    async def validator():
        calls.append("validated")

    service = GraphProviderSettingsService(
        pool=FakePool(state),
        default_provider="ollama",
        nine_router_validator=validator,
    )

    provider = asyncio.run(service.set_graph_build_provider("9router"))

    assert provider == "9router"
    assert calls == ["validated"]
    assert state["rows"]["graph_build_provider"]["value"] == "9router"


def test_set_graph_build_provider_keeps_existing_value_when_nine_router_validation_fails():
    from backend.core.app_settings import GraphProviderSettingsService

    state = {"rows": {"graph_build_provider": {"key": "graph_build_provider", "value": "ollama"}}}

    async def validator():
        raise RuntimeError("9router local proxy is not reachable")

    service = GraphProviderSettingsService(
        pool=FakePool(state),
        default_provider="ollama",
        nine_router_validator=validator,
    )

    with pytest.raises(RuntimeError, match="9router local proxy is not reachable"):
        asyncio.run(service.set_graph_build_provider("9router"))

    assert state["rows"]["graph_build_provider"]["value"] == "ollama"
```

- [ ] **Step 2: Run the focused settings tests to verify they fail**

Run:

```bash
pytest backend/tests/test_graph_provider_settings.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `backend.core.app_settings` because the settings service does not exist yet.

- [ ] **Step 3: Implement the settings service, config, and lifespan bootstrap**

Create `backend/core/app_settings.py`:

```python
from __future__ import annotations

import asyncpg
from typing import Awaitable, Callable, Optional

from backend.config import settings


GRAPH_BUILD_PROVIDER_KEY = "graph_build_provider"
SUPPORTED_GRAPH_BUILD_PROVIDERS = ("ollama", "9router")
CREATE_APP_SETTINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""
INSERT_DEFAULT_SETTING_SQL = """
INSERT INTO app_settings (key, value)
VALUES ($1, $2)
ON CONFLICT (key) DO NOTHING
"""
SELECT_SETTING_SQL = "SELECT value FROM app_settings WHERE key = $1"
UPDATE_SETTING_SQL = """
UPDATE app_settings
SET value = $1, updated_at = NOW()
WHERE key = $2
"""

_pool: Optional[asyncpg.Pool] = None
_service: Optional["GraphProviderSettingsService"] = None


async def _default_nine_router_validator() -> None:
    from backend.core.llm_services import validate_nine_router_connection

    await validate_nine_router_connection()


class GraphProviderSettingsService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        default_provider: str,
        nine_router_validator: Callable[[], Awaitable[None]],
    ):
        self.pool = pool
        self.default_provider = default_provider
        self.nine_router_validator = nine_router_validator

    async def ensure_initialized(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(CREATE_APP_SETTINGS_TABLE_SQL)
            await conn.execute(
                INSERT_DEFAULT_SETTING_SQL,
                GRAPH_BUILD_PROVIDER_KEY,
                self.default_provider,
            )

    async def get_graph_build_provider(self) -> str:
        await self.ensure_initialized()
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(SELECT_SETTING_SQL, GRAPH_BUILD_PROVIDER_KEY)
        if row is None:
            return self.default_provider
        return str(row["value"])

    async def set_graph_build_provider(self, provider: str) -> str:
        normalized = provider.strip().lower()
        if normalized not in SUPPORTED_GRAPH_BUILD_PROVIDERS:
            raise ValueError(
                f"Unsupported graph build provider: {provider}. "
                f"Supported values: {', '.join(SUPPORTED_GRAPH_BUILD_PROVIDERS)}"
            )

        await self.ensure_initialized()
        if normalized == "9router":
            await self.nine_router_validator()

        async with self.pool.acquire() as conn:
            await conn.execute(UPDATE_SETTING_SQL, normalized, GRAPH_BUILD_PROVIDER_KEY)
        return normalized


async def initialize_graph_provider_settings() -> GraphProviderSettingsService:
    global _pool, _service
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DATABASE,
            min_size=1,
            max_size=2,
        )
    if _service is None:
        _service = GraphProviderSettingsService(
            pool=_pool,
            default_provider=settings.GRAPH_BUILD_PROVIDER_DEFAULT,
            nine_router_validator=_default_nine_router_validator,
        )
    await _service.ensure_initialized()
    return _service


def get_graph_provider_settings_service() -> GraphProviderSettingsService:
    if _service is None:
        raise RuntimeError(
            "GraphProviderSettingsService is not initialized. "
            "Call initialize_graph_provider_settings() during startup first."
        )
    return _service


async def close_graph_provider_settings() -> None:
    global _pool, _service
    if _pool is not None:
        await _pool.close()
    _pool = None
    _service = None
```

Update `backend/config.py`:

```python
    GRAPH_BUILD_PROVIDER_DEFAULT: str = "ollama"

    NINE_ROUTER_BASE_URL: str = "http://host.docker.internal:20128/v1"
    NINE_ROUTER_API_KEY: Optional[str] = None
    NINE_ROUTER_INDEX_MODEL: str = "cc/claude-sonnet-4-20250514"
    NINE_ROUTER_TIMEOUT_SECONDS: int = 60
    NINE_ROUTER_MAX_RETRIES: int = 2
    NINE_ROUTER_RETRY_DELAY_SECONDS: int = 3
```

Update `backend/main.py`:

```python
from backend.core.app_settings import (
    close_graph_provider_settings,
    initialize_graph_provider_settings,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await RAGEngine.initialize()
    await initialize_graph_provider_settings()
    yield
    await close_graph_provider_settings()
    await RAGEngine.finalize()
```

- [ ] **Step 4: Run the focused settings tests again**

Run:

```bash
pytest backend/tests/test_graph_provider_settings.py -v
```

Expected: PASS for all four tests.

- [ ] **Step 5: Commit the settings foundation**

```bash
git add backend/tests/test_graph_provider_settings.py backend/core/app_settings.py backend/config.py backend/main.py
git commit -m "feat: persist global graph provider setting"
```

---

### Task 2: Add 9router Local Indexing and Provider-Aware RAGEngine

**Files:**
- Modify: `backend/tests/test_rag_engine.py`
- Modify: `backend/core/llm_services.py`
- Modify: `backend/core/rag_engine.py`

- [ ] **Step 1: Extend the RAGEngine tests with provider-aware ingest cases**

Append these tests to `backend/tests/test_rag_engine.py`:

```python
def test_initialize_only_builds_query_instance(monkeypatch):
    lightrag_module = types.ModuleType("lightrag")
    lightrag_module.LightRAG = FakeLightRAG

    utils_module = types.ModuleType("lightrag.utils")

    class FakeEmbeddingFuncWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    utils_module.EmbeddingFunc = FakeEmbeddingFuncWrapper

    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    monkeypatch.setitem(sys.modules, "lightrag.utils", utils_module)

    import backend.core.rag_engine as rag_engine

    monkeypatch.setattr(rag_engine.RAGEngine, "_build_embedding_func", staticmethod(lambda: FakeEmbeddingFunc()))
    monkeypatch.setattr(rag_engine, "settings", types.SimpleNamespace(
        EMBEDDING_BACKEND="local",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT=5432,
        POSTGRES_USER="user",
        POSTGRES_PASSWORD="pass",
        POSTGRES_DATABASE="db",
        LIGHTRAG_WORKING_DIR="/tmp",
        LIGHTRAG_MAX_ASYNC=1,
        LIGHTRAG_EMBEDDING_MAX_ASYNC=2,
        LIGHTRAG_EMBEDDING_TIMEOUT=180,
        LIGHTRAG_MAX_PARALLEL_INSERT=1,
        LIGHTRAG_CHUNK_SIZE=800,
        LIGHTRAG_CHUNK_OVERLAP_SIZE=100,
        EMBEDDING_DIM=1024,
        EMBEDDING_MAX_TOKEN_SIZE=384,
        EMBEDDING_MODEL="fake-model",
        LLM_MODEL="gemini-3.1-flash-lite",
        OLLAMA_INDEX_MODEL="qwen2.5:3b",
        OLLAMA_NUM_CTX=32768,
        NINE_ROUTER_INDEX_MODEL="cc/test-model",
        SUMMARY_LANGUAGE="vi",
        ENTITY_TYPES=["person"],
    ))

    rag_engine.RAGEngine._query_instance = None
    rag_engine.RAGEngine._ingest_instances = {}
    FakeLightRAG.init_kwargs = []
    FakeLightRAG.storages_initialized = 0

    asyncio.run(rag_engine.RAGEngine.initialize())

    assert len(FakeLightRAG.init_kwargs) == 1
    assert FakeLightRAG.init_kwargs[0]["llm_model_name"] == "gemini-3.1-flash-lite"
    assert FakeLightRAG.storages_initialized == 1


def test_get_ingest_instance_builds_provider_specific_instances(monkeypatch):
    lightrag_module = types.ModuleType("lightrag")
    lightrag_module.LightRAG = FakeLightRAG

    utils_module = types.ModuleType("lightrag.utils")

    class FakeEmbeddingFuncWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    utils_module.EmbeddingFunc = FakeEmbeddingFuncWrapper

    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    monkeypatch.setitem(sys.modules, "lightrag.utils", utils_module)

    import backend.core.rag_engine as rag_engine

    monkeypatch.setattr(rag_engine.RAGEngine, "_build_embedding_func", staticmethod(lambda: FakeEmbeddingFunc()))
    monkeypatch.setattr(rag_engine, "ollama_index_llm_func", object())
    monkeypatch.setattr(rag_engine, "nine_router_index_llm_func", object())
    monkeypatch.setattr(rag_engine, "settings", types.SimpleNamespace(
        EMBEDDING_BACKEND="local",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT=5432,
        POSTGRES_USER="user",
        POSTGRES_PASSWORD="pass",
        POSTGRES_DATABASE="db",
        LIGHTRAG_WORKING_DIR="/tmp",
        LIGHTRAG_MAX_ASYNC=1,
        LIGHTRAG_EMBEDDING_MAX_ASYNC=2,
        LIGHTRAG_EMBEDDING_TIMEOUT=180,
        LIGHTRAG_MAX_PARALLEL_INSERT=1,
        LIGHTRAG_CHUNK_SIZE=800,
        LIGHTRAG_CHUNK_OVERLAP_SIZE=100,
        EMBEDDING_DIM=1024,
        EMBEDDING_MAX_TOKEN_SIZE=384,
        EMBEDDING_MODEL="fake-model",
        LLM_MODEL="gemini-3.1-flash-lite",
        OLLAMA_INDEX_MODEL="qwen2.5:3b",
        OLLAMA_NUM_CTX=32768,
        NINE_ROUTER_INDEX_MODEL="cc/test-model",
        SUMMARY_LANGUAGE="vi",
        ENTITY_TYPES=["person"],
    ))

    rag_engine.RAGEngine._query_instance = object()
    rag_engine.RAGEngine._ingest_instances = {}
    FakeLightRAG.init_kwargs = []
    FakeLightRAG.storages_initialized = 0

    asyncio.run(rag_engine.RAGEngine.get_ingest_instance("ollama"))
    asyncio.run(rag_engine.RAGEngine.get_ingest_instance("9router"))

    assert len(FakeLightRAG.init_kwargs) == 2
    assert FakeLightRAG.init_kwargs[0]["llm_model_name"] == "qwen2.5:3b"
    assert FakeLightRAG.init_kwargs[0]["llm_model_kwargs"] == {"options": {"num_ctx": 32768}}
    assert FakeLightRAG.init_kwargs[1]["llm_model_name"] == "cc/test-model"
    assert FakeLightRAG.init_kwargs[1]["llm_model_kwargs"] == {}
    assert FakeLightRAG.storages_initialized == 2


def test_get_ingest_instance_rejects_unknown_provider():
    import backend.core.rag_engine as rag_engine

    rag_engine.RAGEngine._ingest_instances = {}

    with pytest.raises(ValueError, match="Unsupported graph build provider"):
        asyncio.run(rag_engine.RAGEngine.get_ingest_instance("gemini"))
```

- [ ] **Step 2: Run the RAGEngine tests to verify the new expectations fail**

Run:

```bash
pytest backend/tests/test_rag_engine.py -v
```

Expected: FAIL because `RAGEngine.initialize()` still creates an ingest instance eagerly and `get_ingest_instance()` does not accept a provider.

- [ ] **Step 3: Implement the 9router wrapper and provider-aware ingest cache**

Add these imports to `backend/core/rag_engine.py`:

```python
from backend.core.llm_services import (
    LocalSentenceTransformerEmbeddingFunc,
    QwenEmbeddingFunc,
    gemini_chat_llm_func,
    ollama_index_llm_func,
    nine_router_index_llm_func,
)
```

Replace the ingest-specific parts of `backend/core/rag_engine.py` with:

```python
class RAGEngine:
    _query_instance = None
    _ingest_instances: dict[str, LightRAG] = {}

    @classmethod
    def _get_ingest_backend_config(cls, provider: str):
        normalized = provider.strip().lower()
        if normalized == "ollama":
            return (
                ollama_index_llm_func,
                settings.OLLAMA_INDEX_MODEL,
                {"options": {"num_ctx": settings.OLLAMA_NUM_CTX}},
            )
        if normalized == "9router":
            return (
                nine_router_index_llm_func,
                settings.NINE_ROUTER_INDEX_MODEL,
                {},
            )
        raise ValueError(
            f"Unsupported graph build provider: {provider}. "
            "Use one of: 'ollama', '9router'."
        )

    @classmethod
    async def initialize(cls):
        cls._set_postgres_env()

        if cls._query_instance is None:
            cls._query_instance = cls._build_rag(
                gemini_chat_llm_func,
                settings.LLM_MODEL,
            )
            await cls._query_instance.initialize_storages()

        return cls._query_instance

    @classmethod
    async def get_ingest_instance(cls, provider: str):
        cls._set_postgres_env()
        normalized = provider.strip().lower()
        if normalized not in cls._ingest_instances:
            llm_func, llm_model_name, llm_model_kwargs = cls._get_ingest_backend_config(normalized)
            cls._ingest_instances[normalized] = cls._build_rag(
                llm_func,
                llm_model_name,
                llm_model_kwargs=llm_model_kwargs,
            )
            await cls._ingest_instances[normalized].initialize_storages()
        return cls._ingest_instances[normalized]

    @classmethod
    async def finalize(cls):
        cls._query_instance = None
        cls._ingest_instances = {}
        print("INFO: RAG Engine connections closed.")
```

Add the 9router client and wrappers to `backend/core/llm_services.py`:

```python
_nine_router_client: Optional[openai.AsyncOpenAI] = None


def get_nine_router_client():
    global _nine_router_client
    if _nine_router_client is None:
        if not settings.NINE_ROUTER_API_KEY:
            raise RuntimeError(
                "NINE_ROUTER_API_KEY is not configured. Set NINE_ROUTER_API_KEY in .env "
                "to enable 9router graph building."
            )
        _nine_router_client = openai.AsyncOpenAI(
            api_key=settings.NINE_ROUTER_API_KEY,
            base_url=settings.NINE_ROUTER_BASE_URL,
        )
    return _nine_router_client


def _build_openai_chat_messages(
    prompt: str,
    system_prompt: str | None = None,
    history: List[dict] | None = None,
):
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    if history:
        for item in history:
            content = str(item.get("content", "")).strip()
            role = str(item.get("role", "user")).strip() or "user"
            if content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


async def validate_nine_router_connection() -> None:
    client = get_nine_router_client()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.NINE_ROUTER_INDEX_MODEL,
                messages=[{"role": "user", "content": "Respond with OK"}],
                temperature=0.0,
                max_tokens=8,
            ),
            timeout=min(settings.NINE_ROUTER_TIMEOUT_SECONDS, 15),
        )
    except Exception as error:
        raise RuntimeError(
            f"9router local proxy is not reachable at {settings.NINE_ROUTER_BASE_URL}: {error}"
        ) from error

    response_text = (
        response.choices[0].message.content
        if response.choices and response.choices[0].message
        else ""
    )
    if not str(response_text).strip():
        raise RuntimeError(
            f"9router local proxy returned an empty validation response at "
            f"{settings.NINE_ROUTER_BASE_URL}"
        )


async def nine_router_index_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs
) -> str:
    client = get_nine_router_client()
    messages = _build_openai_chat_messages(
        prompt,
        system_prompt=_build_ollama_index_system_prompt(system_prompt),
        history=history,
    )
    requested_max_tokens = kwargs.get("max_tokens")
    max_output_tokens = settings.LLM_MAX_TOKENS if requested_max_tokens is None else min(
        int(requested_max_tokens), settings.LLM_MAX_TOKENS
    )

    last_error: Exception | None = None

    for attempt in range(1, settings.NINE_ROUTER_MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.NINE_ROUTER_INDEX_MODEL,
                    messages=messages,
                    temperature=float(kwargs.get("temperature", 0.0)),
                    max_tokens=max_output_tokens,
                    stream=False,
                ),
                timeout=settings.NINE_ROUTER_TIMEOUT_SECONDS,
            )
            response_text = str(response.choices[0].message.content or "").strip()

            if response_text and _looks_malformed_extraction_output(response_text):
                print(
                    f"LLM WARNING: 9router extraction output looked malformed on attempt "
                    f"{attempt}/{settings.NINE_ROUTER_MAX_RETRIES}. Retrying with stricter reminder."
                )
                if attempt >= settings.NINE_ROUTER_MAX_RETRIES:
                    return response_text
                messages[-1]["content"] = (
                    f"{prompt.strip()}\n\n"
                    "FINAL REMINDER: return only exact extraction records with no prose and no extra fields."
                )
                await asyncio.sleep(settings.NINE_ROUTER_RETRY_DELAY_SECONDS)
                continue

            return response_text
        except Exception as error:
            last_error = error
            print(
                f"LLM ERROR: 9router request failed on attempt {attempt}/"
                f"{settings.NINE_ROUTER_MAX_RETRIES}: {error}"
            )
            if attempt < settings.NINE_ROUTER_MAX_RETRIES:
                await asyncio.sleep(settings.NINE_ROUTER_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "9router indexing request failed after retries. "
        f"Last error: {last_error}"
    ) from last_error
```

- [ ] **Step 4: Run the RAGEngine tests again**

Run:

```bash
pytest backend/tests/test_rag_engine.py -v
```

Expected: PASS for the existing embedding warmup test plus the new provider-aware ingest tests.

- [ ] **Step 5: Commit the provider-aware ingest runtime**

```bash
git add backend/tests/test_rag_engine.py backend/core/llm_services.py backend/core/rag_engine.py
git commit -m "feat: add 9router graph build backend"
```

---

### Task 3: Add Settings APIs and Provider-Aware Upload Routing

**Files:**
- Modify: `backend/tests/test_upload_route.py`
- Modify: `backend/api/schemas.py`
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Add failing tests for settings endpoints and upload/provider routing**

Append these tests to `backend/tests/test_upload_route.py`:

```python
def test_get_graph_provider_returns_current_setting(monkeypatch):
    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    class FakeSettingsService:
        async def get_graph_build_provider(self):
            return "ollama"

    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: FakeSettingsService())

    response = asyncio.run(routes.get_graph_provider())

    assert response.provider == "ollama"


def test_update_graph_provider_rejects_unsupported_value(monkeypatch):
    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes
    from backend.api.schemas import GraphProviderUpdateRequest

    class FakeSettingsService:
        async def set_graph_build_provider(self, provider: str):
            raise ValueError("Unsupported graph build provider")

    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: FakeSettingsService())

    with pytest.raises(Exception) as exc_info:
        asyncio.run(routes.update_graph_provider(GraphProviderUpdateRequest(provider="gemini")))

    assert "Unsupported graph build provider" in str(exc_info.value)


def test_upload_uses_persisted_graph_provider(tmp_path, monkeypatch):
    from backend.config import settings

    import fastapi.dependencies.utils as fastapi_utils

    monkeypatch.setattr(fastapi_utils, "ensure_multipart_is_installed", lambda: None)

    import backend.api.routes as routes

    settings.LIGHTRAG_WORKING_DIR = str(tmp_path)
    file = DummyUploadFile("sample.txt", b"xin chao")

    class FakeSettingsService:
        async def get_graph_build_provider(self):
            return "9router"

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

    async def fake_get_ingest_rag_engine(provider: str):
        assert provider == "9router"
        return fake_rag

    monkeypatch.setattr(routes, "document_processor", FakeProcessor())
    monkeypatch.setattr(routes, "get_graph_provider_settings_service", lambda: FakeSettingsService())
    monkeypatch.setattr(routes, "get_ingest_rag_engine", fake_get_ingest_rag_engine)

    response = asyncio.run(routes.upload_file(file))

    assert response.status == "success"
    assert "graph provider: 9router" in response.message
    assert fake_rag.calls[0][1]["file_paths"] == ["sample.txt"]
```

- [ ] **Step 2: Run the upload/settings route tests to verify they fail**

Run:

```bash
pytest backend/tests/test_upload_route.py -v
```

Expected: FAIL because the settings endpoint handlers and provider-aware ingest helper do not exist yet.

- [ ] **Step 3: Implement schemas, routes, and provider-aware upload selection**

Add these models to `backend/api/schemas.py`:

```python
class GraphProviderOption(BaseModel):
    value: str
    label: str


class GraphProviderResponse(BaseModel):
    provider: str


class GraphProviderUpdateRequest(BaseModel):
    provider: str


class GraphProviderUpdateResponse(BaseModel):
    provider: str
    status: str
    message: str


class GraphProviderOptionsResponse(BaseModel):
    options: List[GraphProviderOption]
```

Update the helper section in `backend/api/routes.py`:

```python
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    ComparisonResponse,
    UploadResponse,
    GraphProviderResponse,
    GraphProviderUpdateRequest,
    GraphProviderUpdateResponse,
    GraphProviderOptionsResponse,
    GraphProviderOption,
)


def get_graph_provider_settings_service():
    from backend.core.app_settings import get_graph_provider_settings_service as _get_service

    return _get_service()


async def get_ingest_rag_engine(provider: str):
    from backend.core.rag_engine import RAGEngine

    return await RAGEngine.get_ingest_instance(provider)
```

Add these endpoints to `backend/api/routes.py` before `/upload`:

```python
@router.get("/settings/graph-provider", response_model=GraphProviderResponse)
async def get_graph_provider():
    service = get_graph_provider_settings_service()
    provider = await service.get_graph_build_provider()
    return GraphProviderResponse(provider=provider)


@router.get("/settings/graph-provider/options", response_model=GraphProviderOptionsResponse)
async def get_graph_provider_options():
    return GraphProviderOptionsResponse(
        options=[
            GraphProviderOption(value="ollama", label="Ollama"),
            GraphProviderOption(value="9router", label="9router Local"),
        ]
    )


@router.put("/settings/graph-provider", response_model=GraphProviderUpdateResponse)
async def update_graph_provider(request: GraphProviderUpdateRequest):
    service = get_graph_provider_settings_service()
    try:
        provider = await service.set_graph_build_provider(request.provider)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return GraphProviderUpdateResponse(
        provider=provider,
        status="success",
        message=f"Graph build provider updated to {provider}.",
    )
```

Update the upload flow in `backend/api/routes.py`:

```python
        settings_service = get_graph_provider_settings_service()
        provider = await settings_service.get_graph_build_provider()
        rag = await get_ingest_rag_engine(provider)
        existing_status = await _find_existing_document_status(rag, filename)
```

Replace the upload success message with:

```python
        return UploadResponse(
            filename=filename,
            status="success",
            message=(
                f"File indexed using graph provider: {provider} "
                f"({len(content)} characters, ~{legal_chunk_count} legal sections)"
            )
        )
```

- [ ] **Step 4: Run the route tests again**

Run:

```bash
pytest backend/tests/test_upload_route.py -v
```

Expected: PASS for the existing upload tests and the new graph-provider settings/upload tests.

- [ ] **Step 5: Commit the settings API and upload wiring**

```bash
git add backend/tests/test_upload_route.py backend/api/schemas.py backend/api/routes.py
git commit -m "feat: add graph provider settings api"
```

---

### Task 4: Add the Global Provider Selector to the Upload UI

**Files:**
- Modify: `frontend/src/components/FileUpload.tsx`

- [ ] **Step 1: Capture the current frontend baseline**

Run:

```bash
npm run build --prefix frontend
```

Expected: PASS. This is the smallest available frontend verification because the repository does not currently include a frontend test runner.

- [ ] **Step 2: Replace `FileUpload.tsx` with a provider-aware upload component**

Replace `frontend/src/components/FileUpload.tsx` with:

```tsx
import { useEffect, useState } from 'react'
import { Upload, File, X, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import client from '../api/client'

interface FileUploadProps {
  onSuccess?: () => void
}

interface GraphProviderOption {
  value: string
  label: string
}

export default function FileUpload({ onSuccess }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState('')
  const [provider, setProvider] = useState('ollama')
  const [providerOptions, setProviderOptions] = useState<GraphProviderOption[]>([])
  const [providerLoading, setProviderLoading] = useState(true)
  const [providerSaving, setProviderSaving] = useState(false)
  const [providerError, setProviderError] = useState('')

  useEffect(() => {
    const loadProviderState = async () => {
      try {
        const [providerResponse, optionsResponse] = await Promise.all([
          client.get('/settings/graph-provider'),
          client.get('/settings/graph-provider/options'),
        ])
        setProvider(providerResponse.data.provider)
        setProviderOptions(optionsResponse.data.options)
        setProviderError('')
      } catch (error: any) {
        setProviderError(
          error.response?.data?.detail ||
          error.message ||
          'Failed to load graph provider settings.'
        )
      } finally {
        setProviderLoading(false)
      }
    }

    void loadProviderState()
  }, [])

  const handleProviderChange = async (nextProvider: string) => {
    const previousProvider = provider
    setProvider(nextProvider)
    setProviderSaving(true)
    setProviderError('')

    try {
      const response = await client.put('/settings/graph-provider', {
        provider: nextProvider,
      })
      setProvider(response.data.provider)
      setMessage(response.data.message)
      setStatus('success')
    } catch (error: any) {
      setProvider(previousProvider)
      setProviderError(
        error.response?.data?.detail ||
        error.message ||
        'Failed to update graph provider.'
      )
    } finally {
      setProviderSaving(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setStatus('idle')
      setMessage('')
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setStatus('uploading')
    setMessage('')
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await client.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      if (response.data?.status !== 'success') {
        throw new Error(response.data?.message || 'Upload failed')
      }

      setStatus('success')
      setMessage(response.data.message)
      setFile(null)
      if (onSuccess) onSuccess()
    } catch (error: any) {
      setStatus('error')
      setMessage(
        error.response?.data?.detail ||
        error.message ||
        'Cannot reach backend API. Make sure the backend is running on port 8000.'
      )
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2 rounded-xl border bg-muted/30 p-3">
        <div className="flex items-center justify-between gap-3">
          <label htmlFor="graph-provider" className="text-xs font-semibold text-foreground">
            Graph Build Provider
          </label>
          {(providerLoading || providerSaving) && (
            <span className="flex items-center gap-1 text-[10px] font-medium text-muted-foreground">
              <Loader2 className="w-3 h-3 animate-spin" />
              {providerSaving && provider === '9router' ? 'Checking 9router...' : 'Saving...'}
            </span>
          )}
        </div>

        <select
          id="graph-provider"
          value={provider}
          disabled={providerLoading || providerSaving}
          onChange={(e) => void handleProviderChange(e.target.value)}
          className="w-full rounded-lg border bg-background px-3 py-2 text-xs font-medium text-foreground"
        >
          {providerOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <p className="text-[10px] text-muted-foreground">
          Current provider applies to all future uploads until changed.
        </p>

        {providerError && (
          <div className="flex items-center gap-2 text-[10px] font-bold text-red-600 bg-red-50 p-2 rounded-lg border border-red-100">
            <AlertCircle className="w-3 h-3" />
            <span>{providerError}</span>
          </div>
        )}
      </div>

      {!file ? (
        <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-xl cursor-pointer hover:bg-muted/50 transition-colors group">
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <Upload className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors mb-2" />
            <p className="text-xs text-muted-foreground font-medium">Click to upload legal PDFs or TXTs</p>
          </div>
          <input type="file" className="hidden" accept=".pdf,.txt" onChange={handleFileChange} />
        </label>
      ) : (
        <div className="p-3 border rounded-xl bg-card flex items-center justify-between animate-in zoom-in-95">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="p-2 bg-primary/10 rounded-lg">
              <File className="w-4 h-4 text-primary" />
            </div>
            <span className="text-xs font-medium truncate">{file.name}</span>
          </div>
          <button onClick={() => setFile(null)} className="p-1 hover:bg-muted rounded-full">
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>
      )}

      <p className="text-[10px] text-muted-foreground px-1">
        Uploads will build the graph with: <span className="font-semibold text-foreground">{provider}</span>
      </p>

      {file && status === 'idle' && (
        <button
          onClick={handleUpload}
          className="w-full py-2 bg-primary text-primary-foreground rounded-lg text-xs font-bold hover:opacity-90 transition-all shadow-lg shadow-primary/20"
        >
          Begin Indexing
        </button>
      )}

      {status === 'uploading' && (
        <div className="flex items-center justify-center gap-2 text-xs font-medium text-muted-foreground py-2">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>Building graph with {provider}...</span>
        </div>
      )}

      {status === 'success' && (
        <div className="flex items-center gap-2 text-[10px] font-bold text-emerald-600 bg-emerald-50 p-2 rounded-lg border border-emerald-100">
          <CheckCircle2 className="w-3 h-3" />
          <span>{message}</span>
        </div>
      )}

      {status === 'error' && (
        <div className="flex items-center gap-2 text-[10px] font-bold text-red-600 bg-red-50 p-2 rounded-lg border border-red-100">
          <AlertCircle className="w-3 h-3" />
          <span>{message}</span>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Build the frontend to verify the component compiles**

Run:

```bash
npm run build --prefix frontend
```

Expected: PASS with a production Vite bundle emitted under `frontend/dist`.

- [ ] **Step 4: Manually verify the provider UX in the running app**

Run:

```bash
docker compose up --build frontend backend
```

Expected:

- the sidebar shows a `Graph Build Provider` selector
- changing to `9router` shows `Checking 9router...`
- a failing 9router validation leaves the selector on the old value and shows the backend error
- a successful change affects the next upload without requiring a page reload

- [ ] **Step 5: Commit the UI changes**

```bash
git add frontend/src/components/FileUpload.tsx
git commit -m "feat: add ui graph provider selector"
```

---

### Task 5: Update Docs and Run Full Verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PROJECT_STRUCTURE.md`

- [ ] **Step 1: Update `.env.example` with the new graph-provider and 9router variables**

Add these lines below the Ollama settings in `.env.example`:

```dotenv
GRAPH_BUILD_PROVIDER_DEFAULT=ollama
NINE_ROUTER_BASE_URL=http://host.docker.internal:20128/v1
NINE_ROUTER_API_KEY=your_9router_api_key_here
NINE_ROUTER_INDEX_MODEL=cc/claude-sonnet-4-20250514
NINE_ROUTER_TIMEOUT_SECONDS=60
NINE_ROUTER_MAX_RETRIES=2
NINE_ROUTER_RETRY_DELAY_SECONDS=3
```

- [ ] **Step 2: Update README and architecture docs to describe the new global graph-build setting**

Update `README.md`:

```md
- **Configurable Graph Builder**: Choose `Ollama` or `9router Local` from the UI for future knowledge-graph indexing runs.
```

```md
- **LLM/Embeddings**: Gemini Developer API for chat generation, Ollama or 9router local for LightRAG indexing, Docling for no-OCR PDF text extraction, plus local Vietnamese legal embeddings with `huyydangg/DEk21_hcmute_embedding`
```

```md
### Graph Build Provider

The upload panel includes a global `Graph Build Provider` selector:

- `Ollama` uses the local Ollama indexing model configured by `OLLAMA_*`
- `9router Local` uses the OpenAI-compatible local proxy configured by `NINE_ROUTER_*`

The selected provider is stored in PostgreSQL and applies to all future uploads. Switching to `9router Local` is validated before the backend saves the change.
```

Update `ARCHITECTURE.md`:

```md
| **App Settings Store** | Persists global application settings such as `graph_build_provider` in PostgreSQL and validates 9router before saving it. | `backend/core/app_settings.py` |
| **LLM Wrapper** | Uses Gemini for chat/query generation and provider-specific wrappers for indexing (`ollama` or `9router local`). | `backend/core/llm_services.py` |
```

```md
1. **Ingestion**
   - User uploads a PDF/TXT via `/upload`.
   - UI-selected global `graph_build_provider` is read from PostgreSQL through the settings service.
   - `RAGEngine` selects the matching ingest backend (`ollama` or `9router`) for entity/triple extraction during `ainsert`.
```

Update `PROJECT_STRUCTURE.md`:

```md
│   │   ├─ app_settings.py    # Global app-settings persistence and graph-provider validation
│   │   └─ llm_services.py    # Embedding functions, Gemini chat, Ollama index, 9router index
```

```md
│       ├─ test_graph_provider_settings.py
│       ├─ test_rag_engine.py
│       └─ test_upload_route.py
```

- [ ] **Step 3: Run the full backend test suite**

Run:

```bash
pytest backend/tests -v
```

Expected: PASS across document processing, upload route, rag engine, and graph-provider settings tests.

- [ ] **Step 4: Run the final frontend verification build**

Run:

```bash
npm run build --prefix frontend
```

Expected: PASS.

- [ ] **Step 5: Commit the docs and verification checkpoint**

```bash
git add .env.example README.md ARCHITECTURE.md PROJECT_STRUCTURE.md
git commit -m "docs: describe graph provider selection"
```

---

## Self-Review Coverage Map

- PostgreSQL-backed persistent settings store: Task 1
- `GET/PUT /api/settings/graph-provider` and options endpoint: Task 3
- 9router validation before save: Task 1 and Task 3
- 9router OpenAI-compatible indexing backend: Task 2
- runtime provider-aware ingest selection: Task 2 and Task 3
- upload route uses persisted provider only: Task 3
- global UI selector in upload area: Task 4
- docs and env updates: Task 5

