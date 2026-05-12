import asyncio
import sys
import types


class FakeEmbeddingFunc:
    def __init__(self):
        self.warmup_calls = 0

    def _get_model(self):
        self.warmup_calls += 1
        return object()


class FakeLightRAG:
    init_kwargs = None
    storages_initialized = False

    def __init__(self, **kwargs):
        FakeLightRAG.init_kwargs = kwargs

    async def initialize_storages(self):
        FakeLightRAG.storages_initialized = True


def test_initialize_warms_embedding_model_and_sets_timeout(monkeypatch):
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
        EMBEDDING_DIM=1024,
        EMBEDDING_MODEL="fake-model",
        SUMMARY_LANGUAGE="vi",
        ENTITY_TYPES=["person"],
    ))

    rag_engine.RAGEngine._instance = None
    asyncio.run(rag_engine.RAGEngine.initialize())

    assert FakeLightRAG.init_kwargs["default_embedding_timeout"] == 180
    assert FakeLightRAG.storages_initialized is True


def test_embedding_func_warms_model_on_init(monkeypatch):
    import backend.core.llm_services as llm_services

    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            calls.append(model_name)

    monkeypatch.setattr(llm_services, "settings", types.SimpleNamespace(
        EMBEDDING_MODEL="fake-model",
        EMBEDDING_QUERY_INSTRUCTION="Query: ",
        OPENROUTER_API_KEY="key",
        LLM_MODEL="llm",
    ))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))

    llm_services.VietLegalHarrierEmbeddingFunc()

    assert calls == ["fake-model"]
