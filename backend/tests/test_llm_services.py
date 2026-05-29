from pathlib import Path
import sys
import asyncio

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
