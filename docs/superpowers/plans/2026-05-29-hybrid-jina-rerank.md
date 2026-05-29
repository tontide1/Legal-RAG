# Hybrid Jina Rerank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Jina-based reranking to the `hybrid` query path only, while keeping `naive` explicitly non-reranked and removing the current LightRAG warning from normal app usage.

**Architecture:** Keep a single query-side `LightRAG` instance, add a Jina-backed `rerank_model_func` to that instance, and control rerank behavior per query via `QueryParam.enable_rerank`. Centralize rerank availability and provider wiring in `backend/core/llm_services.py`, force `naive` to opt out in `backend/api/routes.py`, and let `backend/core/hybrid_query.py` opt in only when rerank is configured and available.

**Tech Stack:** Python 3.11, FastAPI, LightRAG, Jina rerank API, pytest, pydantic-settings

---

## File Structure

**Modify:**
- `backend/config.py`
  - Add Jina rerank settings and the `HYBRID_ENABLE_RERANK` toggle.
- `.env.example`
  - Document the new Jina rerank environment variables.
- `backend/core/llm_services.py`
  - Add the Jina rerank wrapper and a small availability helper.
- `backend/core/rag_engine.py`
  - Attach `rerank_model_func` to the query `LightRAG` instance only.
- `backend/api/routes.py`
  - Make `naive` query params explicitly set `enable_rerank=False`.
- `backend/core/hybrid_query.py`
  - Make `hybrid` query params set `enable_rerank` from the app-level availability helper.
- `backend/tests/test_llm_services.py`
  - Add tests for Jina rerank wrapper and availability logic.
- `backend/tests/test_chat_route.py`
  - Add assertions that `naive` query params disable rerank.
- `backend/tests/test_hybrid_query.py`
  - Add tests for `hybrid` rerank enabled and disabled cases.
- `backend/tests/test_rag_engine.py`
  - Add tests for query-side rerank wiring and ingest-side rerank absence.

## Task 1: Add Jina rerank settings and service helpers

**Files:**
- Modify: `backend/config.py`
- Modify: `.env.example`
- Modify: `backend/core/llm_services.py`
- Modify: `backend/tests/test_llm_services.py`

- [ ] **Step 1: Write the failing tests for rerank availability and Jina wrapper**

Append these tests to `backend/tests/test_llm_services.py`:

```python
import asyncio
import types


def test_hybrid_rerank_is_available_only_when_enabled_and_api_key_present(monkeypatch):
    monkeypatch.setattr(
        llm_services,
        "settings",
        types.SimpleNamespace(
            HYBRID_ENABLE_RERANK=True,
            JINA_API_KEY="jina-key",
            JINA_RERANK_MODEL="jina-reranker-v2-base-multilingual",
            JINA_RERANK_BASE_URL="https://api.jina.ai/v1/rerank",
        ),
    )

    assert llm_services.hybrid_rerank_available() is True

    monkeypatch.setattr(
        llm_services,
        "settings",
        types.SimpleNamespace(
            HYBRID_ENABLE_RERANK=True,
            JINA_API_KEY="",
            JINA_RERANK_MODEL="jina-reranker-v2-base-multilingual",
            JINA_RERANK_BASE_URL="https://api.jina.ai/v1/rerank",
        ),
    )

    assert llm_services.hybrid_rerank_available() is False


def test_jina_rerank_func_passes_configured_settings(monkeypatch):
    calls = []

    async def fake_jina_rerank(**kwargs):
        calls.append(kwargs)
        return [{"index": 0, "relevance_score": 0.99}]

    monkeypatch.setattr(
        llm_services,
        "settings",
        types.SimpleNamespace(
            HYBRID_ENABLE_RERANK=True,
            JINA_API_KEY="jina-key",
            JINA_RERANK_MODEL="jina-reranker-v2-base-multilingual",
            JINA_RERANK_BASE_URL="https://api.jina.ai/v1/rerank",
        ),
    )
    monkeypatch.setattr(llm_services, "jina_rerank", fake_jina_rerank, raising=False)

    result = asyncio.run(
        llm_services.jina_rerank_model_func(
            query="xe may",
            documents=["doc-1", "doc-2"],
            top_n=1,
        )
    )

    assert result == [{"index": 0, "relevance_score": 0.99}]
    assert calls == [
        {
            "query": "xe may",
            "documents": ["doc-1", "doc-2"],
            "top_n": 1,
            "api_key": "jina-key",
            "model": "jina-reranker-v2-base-multilingual",
            "base_url": "https://api.jina.ai/v1/rerank",
        }
    ]
```

- [ ] **Step 2: Run the llm service tests to verify they fail**

Run:

```bash
pytest backend/tests/test_llm_services.py -q
```

Expected: FAIL with missing attributes such as `hybrid_rerank_available`, `jina_rerank_model_func`, or missing import of `jina_rerank`.

- [ ] **Step 3: Add the new settings and Jina rerank helpers**

Update `backend/config.py` with these fields near the existing model settings:

```python
    JINA_API_KEY: Optional[str] = None
    JINA_RERANK_MODEL: str = "jina-reranker-v2-base-multilingual"
    JINA_RERANK_BASE_URL: str = "https://api.jina.ai/v1/rerank"
    HYBRID_ENABLE_RERANK: bool = True
```

Update `.env.example` by adding:

```env
# Hybrid Rerank Settings
JINA_API_KEY=your_jina_api_key_here
JINA_RERANK_MODEL=jina-reranker-v2-base-multilingual
JINA_RERANK_BASE_URL=https://api.jina.ai/v1/rerank
HYBRID_ENABLE_RERANK=True
```

Update `backend/core/llm_services.py` imports and helpers:

```python
from lightrag.rerank import jina_rerank
```

```python
def hybrid_rerank_available() -> bool:
    return bool(settings.HYBRID_ENABLE_RERANK and (settings.JINA_API_KEY or "").strip())


async def jina_rerank_model_func(query: str, documents: List[str], top_n: int | None = None):
    return await jina_rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        api_key=settings.JINA_API_KEY,
        model=settings.JINA_RERANK_MODEL,
        base_url=settings.JINA_RERANK_BASE_URL,
    )
```

- [ ] **Step 4: Run the llm service tests to verify they pass**

Run:

```bash
pytest backend/tests/test_llm_services.py -q
```

Expected: PASS with the existing prompt test plus the two new rerank tests.

- [ ] **Step 5: Commit the settings and service-helper work**

Run:

```bash
git add backend/config.py .env.example backend/core/llm_services.py backend/tests/test_llm_services.py
git commit -m "feat: add jina rerank service helpers"
```

## Task 2: Make `naive` explicitly disable rerank

**Files:**
- Modify: `backend/api/routes.py`
- Modify: `backend/tests/test_chat_route.py`

- [ ] **Step 1: Write the failing chat-route test**

Add this test to `backend/tests/test_chat_route.py` after the helper-loading code:

```python
def test_build_query_param_disables_rerank_for_naive(monkeypatch):
    routes, ChatRequest = _load_chat_route_modules(monkeypatch)

    request = ChatRequest(
        message="What is the penalty?",
        history=[{"role": "user", "content": "Earlier context"}],
        stream=False,
    )

    param = routes._build_query_param(request, mode="naive")

    assert param.mode == "naive"
    assert param.enable_rerank is False
```

Update the fake `QueryParam` class at the top of the same file so it can store arbitrary keyword arguments:

```python
    class FakeQueryParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
```

- [ ] **Step 2: Run the chat-route test to verify it fails**

Run:

```bash
pytest backend/tests/test_chat_route.py::test_build_query_param_disables_rerank_for_naive -v
```

Expected: FAIL because `_build_query_param()` does not currently set `enable_rerank`.

- [ ] **Step 3: Implement explicit `naive` rerank opt-out**

Update `_build_query_param()` in `backend/api/routes.py` to pass `enable_rerank` by mode:

```python
def _build_query_param(request: ChatRequest, mode: str, stream: bool = False):
    from lightrag import QueryParam

    return QueryParam(
        mode=mode,
        stream=stream,
        conversation_history=request.history,
        enable_rerank=False if mode == "naive" else None,
    )
```

Then tighten it so only meaningful values are passed:

```python
def _build_query_param(request: ChatRequest, mode: str, stream: bool = False):
    from lightrag import QueryParam

    kwargs = {
        "mode": mode,
        "stream": stream,
        "conversation_history": request.history,
    }
    if mode == "naive":
        kwargs["enable_rerank"] = False

    return QueryParam(**kwargs)
```

- [ ] **Step 4: Run the focused chat-route tests to verify they pass**

Run:

```bash
pytest backend/tests/test_chat_route.py -q
```

Expected: PASS, including the new assertion that the `naive` path disables rerank while existing streaming tests stay green.

- [ ] **Step 5: Commit the `naive` rerank behavior**

Run:

```bash
git add backend/api/routes.py backend/tests/test_chat_route.py
git commit -m "fix: disable rerank for naive queries"
```

## Task 3: Make `hybrid` opt into rerank only when available

**Files:**
- Modify: `backend/core/hybrid_query.py`
- Modify: `backend/tests/test_hybrid_query.py`

- [ ] **Step 1: Write the failing hybrid-query tests**

Add these tests to `backend/tests/test_hybrid_query.py`:

```python
def test_run_hybrid_query_enables_rerank_when_available(monkeypatch):
    hybrid_query = _load_module(monkeypatch)

    monkeypatch.setattr(hybrid_query, "extract_hybrid_intent", lambda query, history: hybrid_query.HybridIntent([], [], [], []))
    monkeypatch.setattr(hybrid_query, "select_anchor_candidate", lambda data, query, intent: None)
    monkeypatch.setattr(hybrid_query, "hybrid_rerank_available", lambda: True, raising=False)

    class FakeRAG:
        def __init__(self):
            self.calls = []

        async def aquery_data(self, query, param):
            self.calls.append((query, param))
            return {"data": {"entities": [], "relationships": [], "chunks": [], "references": []}}

    rag = FakeRAG()
    asyncio.run(hybrid_query.run_hybrid_query(rag, "Điều 9 quy định gì?", []))

    _, param = rag.calls[0]
    assert param.enable_rerank is True


def test_run_hybrid_query_disables_rerank_when_unavailable(monkeypatch):
    hybrid_query = _load_module(monkeypatch)

    monkeypatch.setattr(hybrid_query, "extract_hybrid_intent", lambda query, history: hybrid_query.HybridIntent([], [], [], []))
    monkeypatch.setattr(hybrid_query, "select_anchor_candidate", lambda data, query, intent: None)
    monkeypatch.setattr(hybrid_query, "hybrid_rerank_available", lambda: False, raising=False)

    class FakeRAG:
        def __init__(self):
            self.calls = []

        async def aquery_data(self, query, param):
            self.calls.append((query, param))
            return {"data": {"entities": [], "relationships": [], "chunks": [], "references": []}}

    rag = FakeRAG()
    asyncio.run(hybrid_query.run_hybrid_query(rag, "Điều 9 quy định gì?", []))

    _, param = rag.calls[0]
    assert param.enable_rerank is False
```

- [ ] **Step 2: Run the hybrid-query tests to verify they fail**

Run:

```bash
pytest backend/tests/test_hybrid_query.py -q
```

Expected: FAIL because `run_hybrid_query()` does not currently populate `enable_rerank`.

- [ ] **Step 3: Implement hybrid rerank gating**

Import the helper near the top of `backend/core/hybrid_query.py`:

```python
from backend.core.llm_services import gemini_chat_llm_func, hybrid_rerank_available
```

Then update the `QueryParam(...)` construction inside `run_hybrid_query()`:

```python
    rerank_enabled = hybrid_rerank_available()

    result = await rag.aquery_data(
        query,
        param=QueryParam(
            mode="hybrid",
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            conversation_history=trimmed_history,
            history_turns=len(trimmed_history),
            ll_keywords=intent.ll_keywords,
            hl_keywords=intent.hl_keywords,
            include_references=True,
            enable_rerank=rerank_enabled,
        ),
    )
```

Keep the rest of the anchor-first flow unchanged.

- [ ] **Step 4: Run the hybrid-query tests to verify they pass**

Run:

```bash
pytest backend/tests/test_hybrid_query.py -q
```

Expected: PASS, including the new rerank-enabled and rerank-disabled cases.

- [ ] **Step 5: Commit the hybrid rerank gating**

Run:

```bash
git add backend/core/hybrid_query.py backend/tests/test_hybrid_query.py
git commit -m "feat: gate hybrid rerank by jina availability"
```

## Task 4: Wire rerank into the query `LightRAG` instance only

**Files:**
- Modify: `backend/core/rag_engine.py`
- Modify: `backend/tests/test_rag_engine.py`

- [ ] **Step 1: Write the failing RAG engine test**

Extend `backend/tests/test_rag_engine.py` with assertions inside `test_initialize_warms_embedding_model_and_sets_timeout`:

```python
    assert callable(FakeLightRAG.init_kwargs[0]["rerank_model_func"])
    assert FakeLightRAG.init_kwargs[1].get("rerank_model_func") is None
```

Update the monkeypatched imports at the top of `backend/core/rag_engine.py` expectations by ensuring the test settings namespace includes:

```python
        JINA_API_KEY="jina-key",
        JINA_RERANK_MODEL="jina-reranker-v2-base-multilingual",
        JINA_RERANK_BASE_URL="https://api.jina.ai/v1/rerank",
        HYBRID_ENABLE_RERANK=True,
```

- [ ] **Step 2: Run the RAG engine test to verify it fails**

Run:

```bash
pytest backend/tests/test_rag_engine.py::test_initialize_warms_embedding_model_and_sets_timeout -v
```

Expected: FAIL because `_build_rag()` does not currently pass `rerank_model_func`.

- [ ] **Step 3: Implement query-side rerank wiring**

Update the import list in `backend/core/rag_engine.py`:

```python
from backend.core.llm_services import (
    LocalSentenceTransformerEmbeddingFunc,
    QwenEmbeddingFunc,
    gemini_chat_llm_func,
    jina_rerank_model_func,
    ollama_index_llm_func,
)
```

Change `_build_rag()` to accept an optional rerank function:

```python
    @classmethod
    def _build_rag(
        cls,
        llm_func,
        llm_model_name: str,
        llm_model_kwargs: dict | None = None,
        rerank_model_func=None,
    ):
```

Pass that value into `LightRAG(...)`:

```python
            rerank_model_func=rerank_model_func,
```

Wire the instances in `initialize()`:

```python
            cls._query_instance = cls._build_rag(
                gemini_chat_llm_func,
                settings.LLM_MODEL,
                rerank_model_func=jina_rerank_model_func,
            )
```

```python
            cls._ingest_instance = cls._build_rag(
                ollama_index_llm_func,
                settings.OLLAMA_INDEX_MODEL,
                llm_model_kwargs={"options": {"num_ctx": settings.OLLAMA_NUM_CTX}},
                rerank_model_func=None,
            )
```

- [ ] **Step 4: Run the RAG engine tests to verify they pass**

Run:

```bash
pytest backend/tests/test_rag_engine.py -q
```

Expected: PASS, with query-side rerank wiring confirmed and ingest-side rerank absent.

- [ ] **Step 5: Commit the RAG engine wiring**

Run:

```bash
git add backend/core/rag_engine.py backend/tests/test_rag_engine.py
git commit -m "feat: wire jina rerank into query rag"
```

## Task 5: Full verification

**Files:**
- Verify only

- [ ] **Step 1: Run the targeted backend test suite**

Run:

```bash
pytest \
  backend/tests/test_llm_services.py \
  backend/tests/test_chat_route.py \
  backend/tests/test_hybrid_query.py \
  backend/tests/test_rag_engine.py \
  -q
```

Expected: PASS with zero failures.

- [ ] **Step 2: Confirm the working tree only contains intended implementation changes**

Run:

```bash
git status --short
```

Expected: only the rerank-related files above plus any known untracked user data files already present in the repo.

- [ ] **Step 3: If Jina credentials are available locally, verify hybrid no longer requests rerank blindly**

Run a focused local check in the active environment:

```bash
conda run -n RAG python -c "from backend.core.llm_services import hybrid_rerank_available; print(hybrid_rerank_available())"
```

Expected:

- `True` when `HYBRID_ENABLE_RERANK=True` and `JINA_API_KEY` is set
- `False` when either condition is missing

- [ ] **Step 4: Commit any final cleanup**

Run:

```bash
git add backend/config.py .env.example backend/core/llm_services.py backend/core/rag_engine.py backend/api/routes.py backend/core/hybrid_query.py backend/tests/test_llm_services.py backend/tests/test_chat_route.py backend/tests/test_hybrid_query.py backend/tests/test_rag_engine.py
git commit -m "test: finalize hybrid jina rerank coverage"
```

Only create this commit if there are remaining staged changes after Task 4.

## Self-Review

Spec coverage check:

- Jina config and env docs: Task 1
- Jina rerank wrapper: Task 1
- `naive` rerank opt-out: Task 2
- `hybrid` rerank opt-in with fallback to disabled: Task 3
- query-only `LightRAG` rerank wiring: Task 4
- verification: Task 5

Placeholder scan:

- No `TODO`, `TBD`, or implicit “write tests later” steps remain.
- Every code-changing step contains concrete code or assertions.

Type consistency:

- Helper names are consistent across tasks: `hybrid_rerank_available`, `jina_rerank_model_func`, `enable_rerank`.
- The query/ingest split stays in `RAGEngine.initialize()` and is not duplicated elsewhere.
