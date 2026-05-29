from __future__ import annotations

import asyncio

import openai
import asyncpg

from backend.config import settings


GRAPH_BUILD_PROVIDER_KEY = "graph_build_provider"
SUPPORTED_GRAPH_BUILD_PROVIDERS = {"ollama", "9router"}

CREATE_APP_SETTINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

INSERT_DEFAULT_GRAPH_PROVIDER_SQL = """
INSERT INTO app_settings (key, value)
VALUES ($1, $2)
ON CONFLICT (key) DO NOTHING
"""

SELECT_GRAPH_PROVIDER_SQL = "SELECT value FROM app_settings WHERE key = $1"
UPSERT_GRAPH_PROVIDER_SQL = """
INSERT INTO app_settings (key, value)
VALUES ($1, $2)
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
"""

_service: "AppSettingsService" | None = None
_nine_router_client: "openai.AsyncOpenAI" | None = None


def _get_default_graph_build_provider() -> str:
    return getattr(
        settings,
        "GRAPH_BUILD_PROVIDER_DEFAULT",
        getattr(settings, "DEFAULT_GRAPH_BUILD_PROVIDER", "ollama"),
    )


def _normalize_graph_build_provider(provider: str) -> str:
    if not isinstance(provider, str):
        raise ValueError("Unsupported graph_build_provider: value must be a string")

    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_GRAPH_BUILD_PROVIDERS:
        raise ValueError(f"Unsupported graph_build_provider: {provider}")
    return normalized


async def validate_9router_connection() -> None:
    client = _get_nine_router_client()
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


def _get_nine_router_client() -> openai.AsyncOpenAI:
    global _nine_router_client
    if _nine_router_client is None:
        api_key = (settings.NINE_ROUTER_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError(
                "NINE_ROUTER_API_KEY is not configured. Set NINE_ROUTER_API_KEY in .env "
                "to enable 9router graph building."
            )
        _nine_router_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=settings.NINE_ROUTER_BASE_URL,
        )
    return _nine_router_client


class AppSettingsService:
    def __init__(self, database_url: str | None = None):
        self._database_url = database_url or settings.DATABASE_URL
        self._pool: asyncpg.Pool | None = None
        self._schema_ready = False

    async def initialize(self) -> None:
        await self._ensure_pool()
        await self._ensure_schema()
        await self._ensure_default_graph_provider()

    async def close(self) -> None:
        if self._pool is None:
            return

        await self._pool.close()
        self._pool = None
        self._schema_ready = False

    async def get_graph_build_provider(self) -> str:
        await self._ensure_schema()
        await self._ensure_default_graph_provider()

        async with self._pool.acquire() as conn:
            provider = await conn.fetchval(SELECT_GRAPH_PROVIDER_SQL, GRAPH_BUILD_PROVIDER_KEY)

        return _normalize_graph_build_provider(provider)

    async def set_graph_build_provider(self, provider: str) -> str:
        normalized_provider = _normalize_graph_build_provider(provider)

        if normalized_provider == "9router":
            await validate_9router_connection()

        await self._ensure_schema()

        async with self._pool.acquire() as conn:
            await conn.execute(
                UPSERT_GRAPH_PROVIDER_SQL,
                GRAPH_BUILD_PROVIDER_KEY,
                normalized_provider,
            )

        return normalized_provider

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self._database_url)
        return self._pool

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(CREATE_APP_SETTINGS_TABLE_SQL)

        self._schema_ready = True

    async def _ensure_default_graph_provider(self) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                INSERT_DEFAULT_GRAPH_PROVIDER_SQL,
                GRAPH_BUILD_PROVIDER_KEY,
                _normalize_graph_build_provider(_get_default_graph_build_provider()),
            )


async def initialize_graph_provider_settings() -> AppSettingsService:
    global _service
    if _service is None:
        _service = AppSettingsService()
    await _service.initialize()
    return _service


def get_graph_provider_settings_service() -> AppSettingsService:
    if _service is None:
        raise RuntimeError(
            "Graph provider settings service is not initialized. "
            "Call initialize_graph_provider_settings() during startup first."
        )
    return _service


async def close_graph_provider_settings() -> None:
    global _service
    if _service is None:
        return

    await _service.close()
    _service = None
