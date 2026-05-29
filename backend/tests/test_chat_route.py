import asyncio
import json
import sys
import types
from contextlib import contextmanager
from collections import Counter
from pathlib import Path


def _install_fake_lightrag(monkeypatch):
    lightrag_module = types.ModuleType("lightrag")

    class FakeQueryParam:
        def __init__(self, mode, stream=False, conversation_history=None):
            self.mode = mode
            self.stream = stream
            self.conversation_history = conversation_history

    lightrag_module.QueryParam = FakeQueryParam
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)


@contextmanager
def _repo_import_path():
    repo_root = str(Path(__file__).resolve().parents[2])
    sys.path.insert(0, repo_root)
    try:
        yield
    finally:
        try:
            sys.path.remove(repo_root)
        except ValueError:
            pass


def _load_chat_route_modules(monkeypatch):
    _install_fake_lightrag(monkeypatch)
    with _repo_import_path():
        import backend.api.routes as routes
        from backend.api.schemas import ChatRequest
    return routes, ChatRequest


async def _collect_sse_payloads(streaming_response):
    payloads = []
    buffer = ""
    async for chunk in streaming_response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        buffer += chunk

        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            trimmed = frame.strip()
            if not trimmed.startswith("data: "):
                continue
            payloads.append(json.loads(trimmed.removeprefix("data: ").strip()))

    trimmed = buffer.strip()
    if trimmed.startswith("data: "):
        payloads.append(json.loads(trimmed.removeprefix("data: ").strip()))
    return payloads


def test_collect_sse_payloads_handles_split_frames():
    class FakeStreamingResponse:
        def __init__(self, chunks):
            self.body_iterator = self._iterate(chunks)

        async def _iterate(self, chunks):
            for chunk in chunks:
                yield chunk

    response = FakeStreamingResponse(
        [
            'data: {"type": "start", ',
            '"mode": "hybrid"}\n',
            '\n',
            'data: {"type": "chunk", "mode": "hybrid", ',
            '"content": "split"}\n',
            '\n',
            'data: {"type": "done"}\n',
            '\n',
        ]
    )

    payloads = asyncio.run(_collect_sse_payloads(response))

    assert payloads == [
        {"type": "start", "mode": "hybrid"},
        {"type": "chunk", "mode": "hybrid", "content": "split"},
        {"type": "done"},
    ]


def test_chat_comparison_mode_uses_custom_hybrid_runner_and_forwards_raw_message_to_naive(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

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


def test_chat_single_non_streaming_uses_custom_hybrid_runner(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

    hybrid_calls = []
    history = [{"role": "assistant", "content": "Earlier answer"}]

    class FakeRAG:
        async def aquery(self, query, param):
            raise AssertionError("single non-streaming path should not call rag.aquery")

    async def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        hybrid_calls.append((rag, query, request_history, stream))
        return "single-hybrid-response"

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="Explain article 1",
                history=history,
            )
        )
    )

    assert hybrid_calls == [
        (fake_rag, "Explain article 1", history, False)
    ]
    assert response.response == "single-hybrid-response"
    assert response.mode == "hybrid"


def test_chat_single_stream_falls_back_when_hybrid_stream_is_empty(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

    calls = []

    async def empty_stream():
        if False:
            yield "never"

    class FakeRAG:
        async def aquery(self, query, param):
            raise AssertionError("single streaming fallback should use run_hybrid_query")

    async def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        calls.append((rag, query, request_history, stream))
        if stream:
            return empty_stream()
        return "fallback-hybrid-response"

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="Summarize article 2",
                history=[],
                stream=True,
            )
        )
    )
    payloads = asyncio.run(_collect_sse_payloads(response))

    assert calls == [
        (fake_rag, "Summarize article 2", [], True),
        (fake_rag, "Summarize article 2", [], False),
    ]
    assert payloads == [
        {"type": "start", "mode": "hybrid"},
        {"type": "chunk", "mode": "hybrid", "content": "fallback-hybrid-response"},
        {"type": "done"},
    ]


def test_chat_single_stream_returns_error_when_hybrid_fallback_is_empty(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

    calls = []

    async def empty_stream():
        if False:
            yield "never"

    class FakeRAG:
        async def aquery(self, query, param):
            raise AssertionError("single streaming fallback should use run_hybrid_query")

    async def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        calls.append((rag, query, request_history, stream))
        if stream:
            return empty_stream()
        return ""

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="Summarize article 3",
                history=[],
                stream=True,
            )
        )
    )
    payloads = asyncio.run(_collect_sse_payloads(response))

    assert calls == [
        (fake_rag, "Summarize article 3", [], True),
        (fake_rag, "Summarize article 3", [], False),
    ]
    assert payloads == [
        {"type": "start", "mode": "hybrid"},
        {"type": "error", "mode": "hybrid", "message": "hybrid query returned no content."},
        {"type": "done"},
    ]


def test_chat_streaming_comparison_reports_hybrid_error_event(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

    naive_calls = []
    hybrid_calls = []

    async def naive_stream():
        yield "naive chunk"

    class FakeRAG:
        async def aquery(self, query, param):
            naive_calls.append((query, param))
            if param.mode != "naive":
                raise AssertionError("comparison hybrid stream should not call rag.aquery")
            return naive_stream()

    async def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        hybrid_calls.append((rag, query, request_history, stream))
        if stream:
            raise RuntimeError("hybrid stream exploded")
        return "unused"

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="Compare the remedies",
                history=[],
                comparison_mode=True,
                stream=True,
            )
        )
    )
    payloads = asyncio.run(_collect_sse_payloads(response))

    assert len(naive_calls) == 1
    assert hybrid_calls == [(fake_rag, "Compare the remedies", [], True)]
    assert Counter(item["type"] for item in payloads) == Counter({"start": 2, "chunk": 1, "error": 1, "done": 1})
    assert any(item["type"] == "error" and item.get("mode") == "hybrid" for item in payloads)
    assert payloads[-1] == {"type": "done"}


def test_chat_streaming_comparison_outer_exception_is_mode_scoped_and_terminates(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

    class DummyAwaitable:
        def __await__(self):
            if False:
                yield None
            return "unused"

    class FakeRAG:
        def aquery(self, query, param):
            return DummyAwaitable()

    def fake_run_hybrid_query(rag, query, request_history, *, stream=False):
        return DummyAwaitable()

    original_create_task = routes.asyncio.create_task
    created_tasks = []
    create_task_calls = 0

    def exploding_second_create_task(coro, *args, **kwargs):
        nonlocal create_task_calls
        create_task_calls += 1
        if create_task_calls == 1:
            task = original_create_task(coro, *args, **kwargs)
            created_tasks.append(task)
            return task
        coro.close()
        raise RuntimeError("task setup exploded")

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)
    monkeypatch.setattr(routes.asyncio, "create_task", exploding_second_create_task)

    response = asyncio.run(
        routes.chat(
            ChatRequest(
                message="Compare the remedies",
                history=[],
                comparison_mode=True,
                stream=True,
            )
        )
    )
    payloads = asyncio.run(_collect_sse_payloads(response))

    assert payloads == [
        {"type": "error", "mode": "comparison", "message": "task setup exploded"},
        {"type": "done"},
    ]
    assert len(created_tasks) == 1
    assert created_tasks[0].cancelled()


def test_chat_streaming_uses_custom_hybrid_runner_and_emits_sse_chunks(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

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
    payloads = asyncio.run(_collect_sse_payloads(response))
    assert len(calls) == 1
    rag, query, request_history, stream = calls[0]
    assert rag is fake_rag
    assert query == "Summarize article 1"
    assert request_history == history
    assert stream is True
    assert payloads == [
        {"type": "start", "mode": "hybrid"},
        {"type": "chunk", "mode": "hybrid", "content": "streamed chunk"},
        {"type": "done"},
    ]


def test_chat_streaming_comparison_mode_uses_custom_hybrid_runner_and_raw_message_naive_path(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

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
    payloads = asyncio.run(_collect_sse_payloads(response))

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
