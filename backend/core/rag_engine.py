import os
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from backend.core.llm_services import (
    QwenEmbeddingFunc,
    LocalSentenceTransformerEmbeddingFunc,
    deepseek_llm_func,
)
from backend.config import settings

class RAGEngine:
    _instance = None

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

    @classmethod
    async def initialize(cls):
        """Asynchronously initialize the LightRAG storage pools."""
        if cls._instance is None:
            # Initialize custom embedding function
            embedding_func = cls._build_embedding_func()
            
            # Set environment variables for LightRAG Postgres compatibility
            os.environ["POSTGRES_HOST"] = settings.POSTGRES_HOST
            os.environ["POSTGRES_PORT"] = str(settings.POSTGRES_PORT)
            os.environ["POSTGRES_USER"] = settings.POSTGRES_USER
            os.environ["POSTGRES_PASSWORD"] = settings.POSTGRES_PASSWORD
            os.environ["POSTGRES_DATABASE"] = settings.POSTGRES_DATABASE

            # LightRAG initialization with native Postgres storage
            cls._instance = LightRAG(
                working_dir=settings.LIGHTRAG_WORKING_DIR,
                llm_model_func=deepseek_llm_func,
                llm_model_max_async=settings.LIGHTRAG_MAX_ASYNC,
                embedding_func_max_async=settings.LIGHTRAG_EMBEDDING_MAX_ASYNC,
                default_embedding_timeout=settings.LIGHTRAG_EMBEDDING_TIMEOUT,
                max_parallel_insert=settings.LIGHTRAG_MAX_PARALLEL_INSERT,
                chunk_token_size=settings.LIGHTRAG_CHUNK_SIZE,
                chunk_overlap_token_size=settings.LIGHTRAG_CHUNK_OVERLAP_SIZE,
                embedding_func=EmbeddingFunc(
                    embedding_dim=settings.EMBEDDING_DIM,
                    max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                    func=embedding_func,
                    model_name=settings.EMBEDDING_MODEL
                ),
                # Use strings for storage types (LightRAG will instantiate them)
                kv_storage="PGKVStorage",
                vector_storage="PGVectorStorage",
                graph_storage="PGGraphStorage",
                doc_status_storage="PGDocStatusStorage",
                addon_params={
                    "language": settings.SUMMARY_LANGUAGE,
                    "entity_types": settings.ENTITY_TYPES
                }
            )
            # CRITICAL: Initialize Postgres connection pools
            await cls._instance.initialize_storages()
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("RAGEngine not initialized. Call RAGEngine.initialize() first.")
        return cls._instance

rag_engine = RAGEngine()
