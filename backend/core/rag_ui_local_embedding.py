from __future__ import annotations

import importlib

from lightrag.utils import EmbeddingFunc

from backend.config import settings
from backend.core.llm_services import LocalSentenceTransformerEmbeddingFunc


LOCAL_EMBEDDING_MODEL = "huyydangg/DEk21_hcmute_embedding"


def needs_local_rag_ui_embedding_override(binding: str, model: str) -> bool:
    return binding == "openai" and model == LOCAL_EMBEDDING_MODEL


def build_local_embedding_override(
    *,
    local_embedder=None,
    model_name: str,
    embedding_dim: int,
    max_token_size: int,
) -> EmbeddingFunc:
    embedder = local_embedder or LocalSentenceTransformerEmbeddingFunc()

    async def openai_compatible_local_embed(
        texts,
        model=None,
        base_url=None,
        api_key=None,
        embedding_dim=None,
        **kwargs,
    ):
        return await embedder(texts)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=openai_compatible_local_embed,
        model_name=model_name,
    )


def apply_local_embedding_override(
    *,
    binding: str,
    model_name: str,
    embedding_dim: int,
    max_token_size: int,
    local_embedder=None,
) -> bool:
    if not needs_local_rag_ui_embedding_override(binding=binding, model=model_name):
        return False

    openai_module = importlib.import_module("lightrag.llm.openai")
    openai_module.openai_embed = build_local_embedding_override(
        local_embedder=local_embedder,
        model_name=model_name,
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
    )
    return True


def apply_default_local_embedding_override() -> bool:
    return apply_local_embedding_override(
        binding="openai",
        model_name=settings.EMBEDDING_MODEL,
        embedding_dim=settings.EMBEDDING_DIM,
        max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
    )
