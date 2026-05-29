import asyncio
import json
import sys
import types
from collections import Counter


def _install_fake_lightrag(monkeypatch):
    lightrag_module = types.ModuleType("lightrag")

    class FakeQueryParam:
        def __init__(self, mode, stream=False, conversation_history=None):
            self.mode = mode
            self.stream = stream
            self.conversation_history = conversation_history

    lightrag_module.QueryParam = FakeQueryParam
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)


def test_chat_comparison_mode_uses_custom_hybrid_runner_and_forwards_raw_message_to_naive(monkeypatch):
    import backend.api.routes as routes
    from backend.api.schemas import ChatRequest

    _install_fake_lightrag(monkeypatch)

    naive_calls = []
    hybrid_calls = []
    history = [{"role": "user", "content": "Earlier context"}]

    class FakeRAG:
        async def aquery(self, query, param):
            naive_calls.append((query, param))
            if param.mode != "naive":
                raise AssertionError("comparison naive path should be the only rag.aquery call")
            return "naive-response"

    async def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        hybrid_calls.append((rag, query, request_history, stream))
        return "custom-hybrid-response"

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="What is the penalty?",
                history=history,
                comparison_mode=True,
            )
        )
    )

    assert len(naive_calls) == 1
    query, param = naive_calls[0]
    assert query == "What is the penalty?"
    assert param.mode == "naive"
    assert param.stream is False
    assert param.conversation_history == history
    assert hybrid_calls == [
        (fake_rag, "What is the penalty?", history, False)
    ]
    assert response.naive.response == "naive-response"
    assert response.hybrid.response == "custom-hybrid-response"


def test_chat_streaming_uses_custom_hybrid_runner_and_emits_sse_chunks(monkeypatch):
    import backend.api.routes as routes
    from backend.api.schemas import ChatRequest

    _install_fake_lightrag(monkeypatch)

    calls = []
    history = [{"role": "assistant", "content": "Existing answer"}]

    async def hybrid_stream():
        yield "streamed chunk"

    class FakeRAG:
        async def aquery(self, query, param):
            raise AssertionError("single streaming path should not call rag.aquery")

    async def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        calls.append((rag, query, request_history, stream))
        return hybrid_stream()

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="Summarize article 1",
                history=history,
                stream=True,
            )
        )
    )

    async def collect_chunks(streaming_response):
        chunks = []
        async for chunk in streaming_response.body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_chunks(response))

    assert len(calls) == 1
    rag, query, request_history, stream = calls[0]
    assert rag is fake_rag
    assert query == "Summarize article 1"
    assert request_history == history
    assert stream is True
    payloads = [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks]
    assert payloads == [
        {"type": "chunk", "mode": "hybrid", "content": "streamed chunk"},
        {"type": "done"},
    ]


def test_chat_streaming_comparison_mode_uses_custom_hybrid_runner_and_raw_message_naive_path(monkeypatch):
    import backend.api.routes as routes
    from backend.api.schemas import ChatRequest

    _install_fake_lightrag(monkeypatch)

    naive_calls = []
    hybrid_calls = []
    history = [{"role": "assistant", "content": "Existing answer"}]

    async def naive_stream():
        yield "naive chunk"

    async def hybrid_stream():
        yield "hybrid chunk"

    class FakeRAG:
        async def aquery(self, query, param):
            naive_calls.append((query, param))
            if param.mode != "naive":
                raise AssertionError("comparison hybrid stream should not call rag.aquery")
            return naive_stream()

    async def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        hybrid_calls.append((rag, query, request_history, stream))
        return hybrid_stream()

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="Compare the remedies",
                history=history,
                comparison_mode=True,
                stream=True,
            )
        )
    )

    async def collect_chunks(streaming_response):
        chunks = []
        async for chunk in streaming_response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect_chunks(response))
    payloads = [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.strip().startswith("data: ")
    ]

    assert len(naive_calls) == 1
    query, param = naive_calls[0]
    assert query == "Compare the remedies"
    assert param.mode == "naive"
    assert param.stream is True
    assert param.conversation_history == history
    assert hybrid_calls == [
        (fake_rag, "Compare the remedies", history, True)
    ]
    assert Counter(item["type"] for item in payloads) == Counter({"start": 2, "chunk": 2, "done": 1})
    assert sorted(
        (item["mode"], item.get("content"))
        for item in payloads
        if item["type"] == "chunk"
    ) == [("hybrid", "hybrid chunk"), ("naive", "naive chunk")]


def test_chat_request_history_defaults_to_independent_empty_lists():
    from backend.api.schemas import ChatRequest

    first = ChatRequest(message="First")
    second = ChatRequest(message="Second")

    first.history.append({"role": "user", "content": "mutated"})

    assert first.history == [{"role": "user", "content": "mutated"}]
    assert second.history == []
