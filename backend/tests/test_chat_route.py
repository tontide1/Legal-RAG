import asyncio
import json
import sys
import types


def _install_fake_lightrag(monkeypatch):
    lightrag_module = types.ModuleType("lightrag")

    class FakeQueryParam:
        def __init__(self, mode, stream=False, conversation_history=None):
            self.mode = mode
            self.stream = stream
            self.conversation_history = conversation_history

    lightrag_module.QueryParam = FakeQueryParam
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)


def test_chat_comparison_mode_uses_raw_message_for_naive_and_forwards_history(monkeypatch):
    import backend.api.routes as routes
    from backend.api.schemas import ChatRequest

    _install_fake_lightrag(monkeypatch)

    calls = []
    history = [{"role": "user", "content": "Earlier context"}]

    class FakeRAG:
        async def aquery(self, query, param):
            calls.append((query, param))
            return f"{param.mode}-response"

    monkeypatch.setattr(routes, "get_rag_engine", lambda: FakeRAG())

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="What is the penalty?",
                history=history,
                comparison_mode=True,
            )
        )
    )

    assert [query for query, _ in calls] == [
        "What is the penalty?",
        "What is the penalty?",
    ]
    assert [param.mode for _, param in calls] == ["naive", "hybrid"]
    assert [param.stream for _, param in calls] == [False, False]
    assert [param.conversation_history for _, param in calls] == [history, history]
    assert response.naive.response == "naive-response"
    assert response.hybrid.response == "hybrid-response"


def test_chat_streaming_uses_raw_message_and_forwards_history(monkeypatch):
    import backend.api.routes as routes
    from backend.api.schemas import ChatRequest

    _install_fake_lightrag(monkeypatch)

    calls = []
    history = [{"role": "assistant", "content": "Existing answer"}]

    async def hybrid_stream():
        yield "streamed chunk"

    class FakeRAG:
        async def aquery(self, query, param):
            calls.append((query, param))
            return hybrid_stream()

    monkeypatch.setattr(routes, "get_rag_engine", lambda: FakeRAG())

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
    query, param = calls[0]
    assert query == "Summarize article 1"
    assert param.mode == "hybrid"
    assert param.stream is True
    assert param.conversation_history == history
    payloads = [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks]
    assert payloads == [
        {"type": "chunk", "mode": "hybrid", "content": "streamed chunk"},
        {"type": "done"},
    ]


def test_chat_request_history_defaults_to_independent_empty_lists():
    from backend.api.schemas import ChatRequest

    first = ChatRequest(message="First")
    second = ChatRequest(message="Second")

    first.history.append({"role": "user", "content": "mutated"})

    assert first.history == [{"role": "user", "content": "mutated"}]
    assert second.history == []
