import asyncio
import sys
import types

from lightrag.utils import EmbeddingFunc


def test_rag_ui_needs_local_embedding_override_for_repo_model():
    from backend.core.rag_ui_local_embedding import needs_local_rag_ui_embedding_override

    assert (
        needs_local_rag_ui_embedding_override(
            binding="openai",
            model="huyydangg/DEk21_hcmute_embedding",
        )
        is True
    )
    assert (
        needs_local_rag_ui_embedding_override(
            binding="openai",
            model="text-embedding-3-small",
        )
        is False
    )
    assert (
        needs_local_rag_ui_embedding_override(
            binding="ollama",
            model="huyydangg/DEk21_hcmute_embedding",
        )
        is False
    )


def test_build_local_embedding_override_accepts_openai_style_kwargs():
    from backend.core.rag_ui_local_embedding import build_local_embedding_override

    calls = []

    class FakeLocalEmbedder:
        async def __call__(self, texts):
            calls.append(texts)
            return [[0.1, 0.2, 0.3]]

    override = build_local_embedding_override(
        local_embedder=FakeLocalEmbedder(),
        model_name="huyydangg/DEk21_hcmute_embedding",
        embedding_dim=3,
        max_token_size=384,
    )

    assert isinstance(override, EmbeddingFunc)

    result = asyncio.run(
        override.func(
            texts=["Summarize recent commits"],
            model="ignored-by-local-wrapper",
            base_url="https://openrouter.ai/api/v1",
            api_key="ignored",
        )
    )

    assert calls == [["Summarize recent commits"]]
    assert result == [[0.1, 0.2, 0.3]]


def test_apply_local_embedding_override_replaces_lightrag_openai_embed(monkeypatch):
    from backend.core.rag_ui_local_embedding import apply_local_embedding_override

    fake_module = types.ModuleType("lightrag.llm.openai")
    fake_module.openai_embed = "original"
    monkeypatch.setitem(sys.modules, "lightrag.llm.openai", fake_module)

    class FakeLocalEmbedder:
        async def __call__(self, texts):
            return [[0.1, 0.2]]

    applied = apply_local_embedding_override(
        binding="openai",
        model_name="huyydangg/DEk21_hcmute_embedding",
        embedding_dim=2,
        max_token_size=384,
        local_embedder=FakeLocalEmbedder(),
    )

    assert applied is True
    assert isinstance(fake_module.openai_embed, EmbeddingFunc)
