import os
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from backend.config import settings
from backend.core.llm_services import (
    LocalSentenceTransformerEmbeddingFunc,
    QwenEmbeddingFunc,
    jina_rerank_model_func,
    gemini_chat_llm_func,
    ollama_index_llm_func,
)


class RAGEngine:
    _query_instance = None
    _ingest_instance = None

    @staticmethod
    def _build_embedding_func():
        backend = settings.EMBEDDING_BACKEND.lower()

        if backend == "openrouter":
            return QwenEmbeddingFunc()
        if backend in {"sentence_transformers", "huggingface_local", "local"}:
            return LocalSentenceTransformerEmbeddingFunc()

        raise ValueError(
            f"Unsupported EMBEDDING_BACKEND='{settings.EMBEDDING_BACKEND}'. "
            "Use one of: 'openrouter', 'sentence_transformers', 'huggingface_local', 'local'."
        )

    @staticmethod
    def _set_postgres_env():
        os.environ["POSTGRES_HOST"] = settings.POSTGRES_HOST
        os.environ["POSTGRES_PORT"] = str(settings.POSTGRES_PORT)
        os.environ["POSTGRES_USER"] = settings.POSTGRES_USER
        os.environ["POSTGRES_PASSWORD"] = settings.POSTGRES_PASSWORD
        os.environ["POSTGRES_DATABASE"] = settings.POSTGRES_DATABASE

    @classmethod
    def _build_rag(
        cls,
        llm_func,
        llm_model_name: str,
        llm_model_kwargs: dict | None = None,
        rerank_model_func=None,
    ):
        embedding_func = cls._build_embedding_func()

        return LightRAG(
            working_dir=settings.LIGHTRAG_WORKING_DIR,
            llm_model_func=llm_func,
            llm_model_name=llm_model_name,
            rerank_model_func=rerank_model_func,
            llm_model_max_async=settings.LIGHTRAG_MAX_ASYNC,
            llm_model_kwargs=llm_model_kwargs or {},
            embedding_func_max_async=settings.LIGHTRAG_EMBEDDING_MAX_ASYNC,
            default_embedding_timeout=settings.LIGHTRAG_EMBEDDING_TIMEOUT,
            max_parallel_insert=settings.LIGHTRAG_MAX_PARALLEL_INSERT,
            chunk_token_size=settings.LIGHTRAG_CHUNK_SIZE,
            chunk_overlap_token_size=settings.LIGHTRAG_CHUNK_OVERLAP_SIZE,
            embedding_func=EmbeddingFunc(
                embedding_dim=settings.EMBEDDING_DIM,
                max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                func=embedding_func,
                model_name=settings.EMBEDDING_MODEL,
            ),
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            graph_storage="PGGraphStorage",
            doc_status_storage="PGDocStatusStorage",
            addon_params={
                "language": settings.SUMMARY_LANGUAGE,
                "entity_types": settings.ENTITY_TYPES,
            },
        )

    @classmethod
    async def initialize(cls):
        """Initialize separate LightRAG instances for query and ingest."""
        cls._set_postgres_env()

        if cls._query_instance is None:
            cls._query_instance = cls._build_rag(
                gemini_chat_llm_func,
                settings.LLM_MODEL,
                rerank_model_func=jina_rerank_model_func,
            )
            await cls._query_instance.initialize_storages()

        if cls._ingest_instance is None:
            cls._ingest_instance = cls._build_rag(
                ollama_index_llm_func,
                settings.OLLAMA_INDEX_MODEL,
                llm_model_kwargs={"options": {"num_ctx": settings.OLLAMA_NUM_CTX}},
            )
            await cls._ingest_instance.initialize_storages()

        return cls._query_instance

    @classmethod
    async def finalize(cls):
        cls._query_instance = None
        cls._ingest_instance = None
        print("INFO: RAG Engine connections closed.")

    @classmethod
    def get_query_instance(cls):
        if cls._query_instance is None:
            raise RuntimeError("RAGEngine query instance not initialized. Call RAGEngine.initialize() first.")
        return cls._query_instance

    @classmethod
    def get_ingest_instance(cls):
        if cls._ingest_instance is None:
            raise RuntimeError("RAGEngine ingest instance not initialized. Call RAGEngine.initialize() first.")
        return cls._ingest_instance

    @classmethod
    def get_instance(cls):
        return cls.get_query_instance()


rag_engine = RAGEngine()
