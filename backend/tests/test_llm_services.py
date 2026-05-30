from pathlib import Path
import sys
import asyncio
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.core.llm_services as llm_services


def test_build_ollama_index_system_prompt_includes_traffic_law_guidance():
    prompt = llm_services._build_ollama_index_system_prompt("Extract graph records.")

    assert prompt.startswith("TRAFFIC-LAW GRAPH EXTRACTION")
    assert "Preserve exact Vietnamese legal phrases" in prompt
    assert "Do not paraphrase or normalize legal citations" in prompt
    for entity_type in llm_services.settings.ENTITY_TYPES:
        assert f"- {entity_type}" in prompt
    assert "STRICT OUTPUT RULES:" in prompt
    assert "Extract graph records." in prompt


def test_hybrid_rerank_available_depends_on_flag_and_api_key(monkeypatch):
    monkeypatch.setattr(llm_services.settings, "HYBRID_ENABLE_RERANK", True)
    monkeypatch.setattr(llm_services.settings, "JINA_API_KEY", "  secret-key  ")

    assert llm_services.hybrid_rerank_available() is True

    monkeypatch.setattr(llm_services.settings, "JINA_API_KEY", "   ")

    assert llm_services.hybrid_rerank_available() is False

    monkeypatch.setattr(llm_services.settings, "JINA_API_KEY", "secret-key")
    monkeypatch.setattr(llm_services.settings, "HYBRID_ENABLE_RERANK", False)

    assert llm_services.hybrid_rerank_available() is False


def test_jina_rerank_model_func_forwards_configured_settings(monkeypatch):
    captured = {}

    async def fake_jina_rerank(**kwargs):
        captured.update(kwargs)
        return [{"index": 0, "relevance_score": 0.99}]

    monkeypatch.setattr(llm_services, "jina_rerank", fake_jina_rerank)
    monkeypatch.setattr(llm_services.settings, "JINA_API_KEY", "api-key")
    monkeypatch.setattr(llm_services.settings, "JINA_RERANK_MODEL", "custom-model")
    monkeypatch.setattr(llm_services.settings, "JINA_RERANK_BASE_URL", "https://example.com/rerank")

    result = asyncio.run(
        llm_services.jina_rerank_model_func(
            query="What is the rule?",
            documents=["doc-a", "doc-b"],
            top_n=3,
        )
    )

    assert result == [{"index": 0, "relevance_score": 0.99}]
    assert captured == {
        "query": "What is the rule?",
        "documents": ["doc-a", "doc-b"],
        "top_n": 3,
        "api_key": "api-key",
        "model": "custom-model",
        "base_url": "https://example.com/rerank",
    }


def test_validate_nine_router_connection_accepts_reasoning_only_response(monkeypatch):
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

    monkeypatch.setattr(llm_services.settings, "NINE_ROUTER_BASE_URL", "https://router.example")
    monkeypatch.setattr(llm_services.settings, "NINE_ROUTER_INDEX_MODEL", "nvidia/minimaxai/minimax-m2.7")
    monkeypatch.setattr(llm_services.settings, "NINE_ROUTER_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr(llm_services, "get_nine_router_client", lambda: FakeClient())

    asyncio.run(llm_services.validate_nine_router_connection())


def test_nine_router_index_llm_func_returns_reasoning_content_when_content_is_empty(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=None,
                            model_extra={"reasoning_content": "EXTRACTED"},
                        )
                    )
                ]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm_services.settings, "NINE_ROUTER_INDEX_MODEL", "nvidia/minimaxai/minimax-m2.7")
    monkeypatch.setattr(llm_services.settings, "NINE_ROUTER_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr(llm_services.settings, "NINE_ROUTER_MAX_RETRIES", 1)
    monkeypatch.setattr(llm_services.settings, "LLM_MAX_TOKENS", 1024)
    monkeypatch.setattr(llm_services, "get_nine_router_client", lambda: FakeClient())

    result = asyncio.run(llm_services.nine_router_index_llm_func("Prompt"))

    assert result == "EXTRACTED"


def test_nine_router_index_llm_func_caps_max_tokens(monkeypatch):
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="EXTRACTED",
                            model_extra={},
                        )
                    )
                ]
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm_services.settings, "NINE_ROUTER_INDEX_MODEL", "nvidia/minimaxai/minimax-m2.7")
    monkeypatch.setattr(llm_services.settings, "LLM_MAX_TOKENS", 128000)
    monkeypatch.setattr(llm_services, "get_nine_router_client", lambda: FakeClient())

    result = asyncio.run(llm_services.nine_router_index_llm_func("Prompt"))

    assert result == "EXTRACTED"
    assert captured["max_tokens"] == 4096
