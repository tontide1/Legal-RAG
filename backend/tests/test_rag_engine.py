import asyncio
import sys
import types

from backend.config import settings as app_settings


class FakeEmbeddingFunc:
    def __init__(self):
        self.warmup_calls = 0

    def _get_model(self):
        self.warmup_calls += 1
        return object()


class FakeLightRAG:
    init_kwargs = []
    storages_initialized = 0

    def __init__(self, **kwargs):
        FakeLightRAG.init_kwargs.append(kwargs)

    async def initialize_storages(self):
        FakeLightRAG.storages_initialized += 1


def test_initialize_warms_embedding_model_and_sets_timeout(monkeypatch):
    lightrag_module = types.ModuleType("lightrag")
    lightrag_module.__path__ = []
    lightrag_module.LightRAG = FakeLightRAG

    rerank_module = types.ModuleType("lightrag.rerank")
    rerank_module.jina_rerank = lambda *args, **kwargs: None

    utils_module = types.ModuleType("lightrag.utils")

    class FakeEmbeddingFuncWrapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    utils_module.EmbeddingFunc = FakeEmbeddingFuncWrapper

    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)
    monkeypatch.setitem(sys.modules, "lightrag.rerank", rerank_module)
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
        LLM_MAX_TOKENS=1024,
        OLLAMA_INDEX_MODEL="qwen2.5:3b",
        OLLAMA_NUM_CTX=32768,
        SUMMARY_LANGUAGE="vi",
        ENTITY_TYPES=app_settings.ENTITY_TYPES,
    ))

    rag_engine.RAGEngine._query_instance = None
    rag_engine.RAGEngine._ingest_instance = None
    FakeLightRAG.init_kwargs = []
    FakeLightRAG.storages_initialized = 0
    asyncio.run(rag_engine.RAGEngine.initialize())

    assert len(FakeLightRAG.init_kwargs) == 2
    assert FakeLightRAG.init_kwargs[0]["default_embedding_timeout"] == 180
    assert callable(FakeLightRAG.init_kwargs[0]["rerank_model_func"])
    assert FakeLightRAG.init_kwargs[0]["llm_model_kwargs"] == {}
    assert FakeLightRAG.init_kwargs[0]["addon_params"]["entity_types"] == app_settings.ENTITY_TYPES
    assert FakeLightRAG.init_kwargs[1]["llm_model_name"] == "qwen2.5:3b"
    assert FakeLightRAG.init_kwargs[1].get("rerank_model_func") is None
    assert FakeLightRAG.init_kwargs[1]["llm_model_kwargs"] == {"options": {"num_ctx": 32768}}
    assert FakeLightRAG.init_kwargs[1]["addon_params"]["entity_types"] == app_settings.ENTITY_TYPES
    assert FakeLightRAG.storages_initialized == 2


def test_embedding_func_warms_model_on_init(monkeypatch):
    import backend.core.llm_services as llm_services

    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None):
            calls.append((model_name, device))

    monkeypatch.setattr(llm_services, "settings", types.SimpleNamespace(
        EMBEDDING_MODEL="fake-model",
        EMBEDDING_DEVICE="cpu",
        EMBEDDING_QUERY_INSTRUCTION="Query: ",
        OPENROUTER_API_KEY="key",
        LLM_MODEL="llm",
    ))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))

    llm_services.LocalSentenceTransformerEmbeddingFunc()._get_model()

    assert calls == [("fake-model", "cpu")]
