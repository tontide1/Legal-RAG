# Hybrid Anchor-First Traffic-Law Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `hybrid` into an anchor-first synthesis mode for Vietnamese traffic-law questions while keeping the frontend comparison experience as `naive vs hybrid`.

**Architecture:** Keep `naive` on the current LightRAG vector path, but clean up its request plumbing so retrieval uses the raw query plus conversation history. Build a new backend `hybrid` orchestration layer on top of `LightRAG.aquery_data(...)`: extract traffic-law intent, retrieve structured KG data, score one primary `Điều khoản` anchor, bucket supporting chunks by scope/conditions/responsibilities/violations, then generate the final answer with Gemini from the assembled anchor-first context. Expand the ingest ontology and indexing guidance so the rebuilt graph contains the legal signals the new `hybrid` strategy needs.

**Tech Stack:** Python 3.11, FastAPI, LightRAG, Gemini, Ollama, pytest, React 18, TypeScript, Vite.

---

## File Structure

- Create: `backend/core/hybrid_query.py`
  - Own the anchor-first `hybrid` strategy: intent extraction, retrieval overrides, anchor scoring, expansion bucketing, context assembly, and Gemini answer generation.

- Create: `backend/tests/test_chat_route.py`
  - Cover raw-query handling, conversation-history forwarding, and the `hybrid` route integration in both streaming and non-streaming modes.

- Create: `backend/tests/test_hybrid_query.py`
  - Cover anchor selection, expansion bucketing, context ordering, and insufficient-context behavior using synthetic LightRAG retrieval payloads.

- Create: `backend/tests/test_llm_services.py`
  - Cover the new traffic-law indexing guidance appended to the Ollama graph-build system prompt.

- Create: `docs/hybrid-benchmark.md`
  - Capture the synthesis benchmark set, expected anchor behavior, and manual re-index verification checklist.

- Modify: `backend/api/schemas.py`
  - Use safe list defaults for history and sources.

- Modify: `backend/api/routes.py`
  - Stop polluting retrieval queries with formatting instructions, forward conversation history, keep `naive` on LightRAG `aquery`, and route `hybrid` through the new anchor-first service.

- Modify: `backend/config.py`
  - Expand `ENTITY_TYPES` for traffic-law indexing and add small config knobs for hybrid ranking/history limits.

- Modify: `backend/core/llm_services.py`
  - Append traffic-law entity guidance to the Ollama indexing system prompt and expose any helper needed by `hybrid_query.py`.

- Modify: `backend/core/rag_engine.py`
  - Continue to pass `settings.ENTITY_TYPES` into LightRAG `addon_params`, now with the traffic-law-expanded ontology.

- Modify: `backend/tests/test_rag_engine.py`
  - Assert the new ontology reaches LightRAG initialization.

- Modify: `frontend/src/components/ChatInterface.tsx`
  - Send recent conversation history with each `/chat` request, using the current assistant-side `hybrid` answer as the persisted assistant turn when comparison mode is active.

- Modify: `.env.example`, `README.md`, `ARCHITECTURE.md`, `PROJECT_STRUCTURE.md`
  - Document the traffic-law ontology, the new `hybrid` strategy, the re-index requirement, and the new config knobs.

---

### Task 1: Clean Up Chat Retrieval Queries And History Plumbing

**Files:**
- Create: `backend/tests/test_chat_route.py`
- Modify: `backend/api/schemas.py`
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Write the failing backend route tests**

Create `backend/tests/test_chat_route.py`:

```python
import asyncio

from backend.api.schemas import ChatRequest


class FakeRAG:
    def __init__(self):
        self.calls = []

    async def aquery(self, query, param=None, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "param": param,
                "system_prompt": system_prompt,
            }
        )
        return f"{param.mode}:ok"


async def _fake_hybrid_runner(rag, query, history, *, stream=False):
    return "hybrid:stub"


def test_chat_comparison_keeps_naive_retrieval_on_raw_query(monkeypatch):
    import backend.api.routes as routes

    fake_rag = FakeRAG()
    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", _fake_hybrid_runner, raising=False)

    request = ChatRequest(
        message="Ai chịu trách nhiệm thực hiện cứu hộ giao thông đường bộ?",
        history=[
            {"role": "user", "content": "Điều 79 nói về gì?"},
            {"role": "assistant", "content": "Điều 79 nói về dịch vụ cứu hộ giao thông đường bộ."},
        ],
        comparison_mode=True,
        stream=False,
    )

    response = asyncio.run(routes.chat(request))

    assert response.naive.response == "naive:ok"
    assert response.hybrid.response == "hybrid:stub"
    assert len(fake_rag.calls) == 1
    assert fake_rag.calls[0]["query"] == "Ai chịu trách nhiệm thực hiện cứu hộ giao thông đường bộ?"
    assert fake_rag.calls[0]["param"].conversation_history == request.history


def test_chat_request_history_defaults_to_empty_list():
    first = ChatRequest(message="Xin chào")
    second = ChatRequest(message="Tiếp tục")

    first.history.append({"role": "user", "content": "A"})

    assert second.history == []
```

- [ ] **Step 2: Run the focused route tests to verify they fail**

Run:

```bash
pytest backend/tests/test_chat_route.py -v
```

Expected: FAIL because `backend/api/routes.py` still appends `STRICT INSTRUCTION` into the retrieval query and does not populate `conversation_history`.

- [ ] **Step 3: Implement raw-query usage and history forwarding**

Update `backend/api/schemas.py` to use safe defaults:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    stream: Optional[bool] = False
    comparison_mode: Optional[bool] = False


class ChatResponse(BaseModel):
    response: str
    mode: str = "hybrid"
    sources: List[Dict[str, Any]] = Field(default_factory=list)
```

Update `backend/api/routes.py` to stop building `full_query` and instead create a small helper:

```python
def _build_query_param(mode: str, request: ChatRequest, *, stream: bool = False):
    from lightrag import QueryParam

    return QueryParam(
        mode=mode,
        stream=stream,
        conversation_history=request.history,
    )
```

Then replace the existing calls:

```python
naive_response = await rag.aquery(
    request.message,
    param=_build_query_param("naive", request),
)
hybrid_response = await rag.aquery(
    request.message,
    param=_build_query_param("hybrid", request),
)
```

and:

```python
generator = await rag.aquery(
    request.message,
    param=_build_query_param("hybrid", request, stream=True),
)
```

At this stage keep the current LightRAG-driven `hybrid` path intact; Task 4 will swap in the anchor-first orchestration.

- [ ] **Step 4: Re-run the route tests**

Run:

```bash
pytest backend/tests/test_chat_route.py -v
```

Expected: PASS for both route tests.

- [ ] **Step 5: Commit**

```bash
git add backend/api/schemas.py backend/api/routes.py backend/tests/test_chat_route.py
git commit -m "feat: clean chat query handling and history plumbing"
```

---

### Task 2: Expand Traffic-Law Ontology And Indexing Guidance

**Files:**
- Create: `backend/tests/test_llm_services.py`
- Modify: `backend/config.py`
- Modify: `backend/core/llm_services.py`
- Modify: `backend/core/rag_engine.py`
- Modify: `backend/tests/test_rag_engine.py`

- [ ] **Step 1: Write failing ontology and prompt-guidance tests**

Create `backend/tests/test_llm_services.py`:

```python
import types


def test_build_ollama_index_system_prompt_includes_traffic_law_guidance(monkeypatch):
    import backend.core.llm_services as llm_services

    monkeypatch.setattr(
        llm_services,
        "settings",
        types.SimpleNamespace(
            ENTITY_TYPES=["Điều khoản", "Ngoại lệ", "Trách nhiệm"],
        ),
    )

    prompt = llm_services._build_ollama_index_system_prompt("Base system prompt")

    assert "TRAFFIC-LAW GRAPH EXTRACTION" in prompt
    assert "- Điều khoản" in prompt
    assert "- Ngoại lệ" in prompt
    assert "- Trách nhiệm" in prompt
```

Append a new assertion to `backend/tests/test_rag_engine.py`:

```python
    assert FakeLightRAG.init_kwargs[0]["addon_params"]["entity_types"] == ["person"]
```

Then replace the fake settings payload so the test uses the target ontology:

```python
        ENTITY_TYPES=[
            "Văn bản pháp luật",
            "Điều khoản",
            "Đối tượng áp dụng",
            "Ngoại lệ",
            "Trách nhiệm",
            "Hành vi vi phạm",
            "Hình thức xử phạt",
        ],
```

and assert the expanded list reaches LightRAG:

```python
    assert FakeLightRAG.init_kwargs[0]["addon_params"]["entity_types"] == [
        "Văn bản pháp luật",
        "Điều khoản",
        "Đối tượng áp dụng",
        "Ngoại lệ",
        "Trách nhiệm",
        "Hành vi vi phạm",
        "Hình thức xử phạt",
    ]
```

- [ ] **Step 2: Run the focused ontology tests to verify they fail**

Run:

```bash
pytest backend/tests/test_llm_services.py backend/tests/test_rag_engine.py -v
```

Expected: FAIL because the indexing system prompt does not yet include traffic-law extraction guidance and the config ontology is still the smaller generic list.

- [ ] **Step 3: Implement the traffic-law ontology and indexing prompt updates**

Expand `backend/config.py`:

```python
    ENTITY_TYPES: list[str] = [
        "Văn bản pháp luật",
        "Điều khoản",
        "Cơ quan ban hành",
        "Đối tượng áp dụng",
        "Thời hạn",
        "Khái niệm pháp lý",
        "Phạm vi áp dụng",
        "Trách nhiệm",
        "Ngoại lệ",
        "Điều kiện áp dụng",
        "Chủ thể có thẩm quyền",
        "Phương tiện giao thông",
        "Người tham gia giao thông",
        "Hành vi bị cấm",
        "Yêu cầu an toàn",
        "Giấy phép / chứng chỉ",
        "Dịch vụ hỗ trợ giao thông",
        "Kết cấu hạ tầng giao thông",
        "Hình thức xử phạt",
        "Hành vi vi phạm",
    ]
    HYBRID_MAX_HISTORY_MESSAGES: int = 8
    HYBRID_TOP_K: int = 20
    HYBRID_CHUNK_TOP_K: int = 12
    HYBRID_ANCHOR_CHUNK_LIMIT: int = 3
    HYBRID_BUCKET_CHUNK_LIMIT: int = 2
```

Update `backend/core/llm_services.py` so `_build_ollama_index_system_prompt(...)` prepends a domain block:

```python
def _build_traffic_law_extraction_guidance() -> str:
    entity_lines = "\n".join(f"- {entity}" for entity in settings.ENTITY_TYPES)
    return (
        "TRAFFIC-LAW GRAPH EXTRACTION:\n"
        "- Prefer exact legal phrases over broad paraphrases.\n"
        "- Keep Điều khoản references stable and easy to retrieve.\n"
        "- Preserve qualifiers that signal scope, conditions, exceptions, responsibilities, violations, and sanctions.\n"
        "- Extract only the following entity types:\n"
        f"{entity_lines}"
    )


def _build_ollama_index_system_prompt(system_prompt: str | None) -> str:
    strict_format_prompt = (
        "STRICT OUTPUT RULES:\n"
        "- Return only records in the exact extraction format requested.\n"
        "- Do not add explanations, markdown, comments, or code fences.\n"
        "- Do not invent extra columns or fields.\n"
        "- If a field contains punctuation, keep it inside the field text instead of splitting it.\n"
        "- If unsure, return fewer well-formed records rather than malformed output."
    )
    domain_prompt = _build_traffic_law_extraction_guidance()

    base_parts = [part for part in [system_prompt and system_prompt.strip(), domain_prompt, strict_format_prompt] if part]
    return "\n\n".join(base_parts)
```

Leave `backend/core/rag_engine.py` structurally unchanged, but keep passing `settings.ENTITY_TYPES` through `addon_params["entity_types"]`.

- [ ] **Step 4: Re-run the ontology tests**

Run:

```bash
pytest backend/tests/test_llm_services.py backend/tests/test_rag_engine.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/core/llm_services.py backend/core/rag_engine.py backend/tests/test_llm_services.py backend/tests/test_rag_engine.py
git commit -m "feat: expand traffic-law ontology for graph indexing"
```

---

### Task 3: Build The Anchor-First Hybrid Query Strategy

**Files:**
- Create: `backend/tests/test_hybrid_query.py`
- Create: `backend/core/hybrid_query.py`

- [ ] **Step 1: Write the failing strategy unit tests**

Create `backend/tests/test_hybrid_query.py`:

```python
from backend.core.hybrid_query import (
    HybridIntent,
    bucket_expansion_chunks,
    build_hybrid_context,
    select_anchor_candidate,
)


SYNTHETIC_DATA = {
    "entities": [
        {
            "entity_name": "Điều 79. Dịch vụ cứu hộ giao thông đường bộ",
            "entity_type": "Điều khoản",
            "description": "Quy định về cứu hộ giao thông đường bộ",
            "source_id": "chunk-79",
            "file_path": "35-2024-qh15.pdf",
            "reference_id": "1",
        },
        {
            "entity_name": "Điều 80. Dịch vụ phần mềm hỗ trợ kết nối vận tải bằng xe ô tô",
            "entity_type": "Điều khoản",
            "description": "Quy định về phần mềm hỗ trợ kết nối vận tải",
            "source_id": "chunk-80",
            "file_path": "35-2024-qh15.pdf",
            "reference_id": "2",
        },
        {
            "entity_name": "Tổ chức, cá nhân kinh doanh dịch vụ cứu hộ giao thông đường bộ",
            "entity_type": "Đối tượng áp dụng",
            "description": "Chủ thể thực hiện dịch vụ cứu hộ",
            "source_id": "chunk-79",
            "file_path": "35-2024-qh15.pdf",
            "reference_id": "1",
        },
        {
            "entity_name": "Không được sử dụng xe cứu hộ để kinh doanh vận tải hàng hoá bằng xe ô tô",
            "entity_type": "Ngoại lệ",
            "description": "Hạn chế sử dụng xe cứu hộ",
            "source_id": "chunk-79",
            "file_path": "35-2024-qh15.pdf",
            "reference_id": "1",
        },
    ],
    "relationships": [
        {
            "src_id": "Điều 79. Dịch vụ cứu hộ giao thông đường bộ",
            "tgt_id": "Tổ chức, cá nhân kinh doanh dịch vụ cứu hộ giao thông đường bộ",
            "description": "Điều 79 áp dụng với tổ chức, cá nhân kinh doanh dịch vụ cứu hộ",
            "file_path": "35-2024-qh15.pdf",
            "reference_id": "1",
        },
        {
            "src_id": "Điều 79. Dịch vụ cứu hộ giao thông đường bộ",
            "tgt_id": "Không được sử dụng xe cứu hộ để kinh doanh vận tải hàng hoá bằng xe ô tô",
            "description": "Điều 79 có giới hạn sử dụng xe cứu hộ",
            "file_path": "35-2024-qh15.pdf",
            "reference_id": "1",
        },
    ],
    "chunks": [
        {
            "chunk_id": "chunk-79",
            "reference_id": "1",
            "file_path": "35-2024-qh15.pdf",
            "content": "Điều 79. Dịch vụ cứu hộ giao thông đường bộ. Tổ chức, cá nhân kinh doanh dịch vụ cứu hộ giao thông đường bộ... Không được sử dụng xe cứu hộ để kinh doanh vận tải hàng hoá bằng xe ô tô.",
        },
        {
            "chunk_id": "chunk-80",
            "reference_id": "2",
            "file_path": "35-2024-qh15.pdf",
            "content": "Điều 80. Dịch vụ phần mềm hỗ trợ kết nối vận tải bằng xe ô tô...",
        },
    ],
}


def test_select_anchor_candidate_prefers_matching_dieu_khoan():
    intent = HybridIntent(
        anchor_terms=["cứu hộ giao thông đường bộ"],
        ll_keywords=["dịch vụ cứu hộ giao thông đường bộ"],
        hl_keywords=["trách nhiệm", "ngoại lệ"],
        focus_buckets=["scope", "conditions", "responsibilities"],
    )

    anchor = select_anchor_candidate(SYNTHETIC_DATA, "Ai chịu trách nhiệm cứu hộ giao thông đường bộ?", intent)

    assert anchor is not None
    assert anchor["entity_name"] == "Điều 79. Dịch vụ cứu hộ giao thông đường bộ"


def test_bucket_expansion_chunks_groups_scope_and_conditions():
    intent = HybridIntent(
        anchor_terms=["cứu hộ giao thông đường bộ"],
        ll_keywords=["dịch vụ cứu hộ giao thông đường bộ"],
        hl_keywords=["đối tượng áp dụng", "ngoại lệ"],
        focus_buckets=["scope", "conditions"],
    )
    anchor = SYNTHETIC_DATA["entities"][0]

    buckets = bucket_expansion_chunks(SYNTHETIC_DATA, anchor, intent)

    assert "scope" in buckets
    assert "conditions" in buckets
    assert buckets["scope"][0]["reference_id"] == "1"
    assert buckets["conditions"][0]["reference_id"] == "1"


def test_build_hybrid_context_orders_sections_and_references():
    intent = HybridIntent(
        anchor_terms=["cứu hộ giao thông đường bộ"],
        ll_keywords=["dịch vụ cứu hộ giao thông đường bộ"],
        hl_keywords=["đối tượng áp dụng", "ngoại lệ"],
        focus_buckets=["scope", "conditions"],
    )
    anchor = SYNTHETIC_DATA["entities"][0]
    buckets = bucket_expansion_chunks(SYNTHETIC_DATA, anchor, intent)

    context = build_hybrid_context(
        query="Ai chịu trách nhiệm và có ngoại lệ gì trong dịch vụ cứu hộ giao thông đường bộ?",
        anchor=anchor,
        buckets=buckets,
    )

    assert "## Anchor Provision" in context
    assert "## Scope And Regulated Subjects" in context
    assert "## Conditions And Exceptions" in context
    assert "## References" in context
```

- [ ] **Step 2: Run the hybrid strategy tests to verify they fail**

Run:

```bash
pytest backend/tests/test_hybrid_query.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.core.hybrid_query'`.

- [ ] **Step 3: Implement the anchor-first strategy module**

Create `backend/core/hybrid_query.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, AsyncIterator

from lightrag import QueryParam

from backend.config import settings
from backend.core.llm_services import gemini_chat_llm_func


FOCUS_BUCKET_LABELS = {
    "scope": {
        "Đối tượng áp dụng",
        "Phạm vi áp dụng",
        "Người tham gia giao thông",
        "Phương tiện giao thông",
    },
    "conditions": {
        "Điều kiện áp dụng",
        "Ngoại lệ",
        "Yêu cầu an toàn",
        "Giấy phép / chứng chỉ",
    },
    "responsibilities": {
        "Trách nhiệm",
        "Chủ thể có thẩm quyền",
    },
    "violations": {
        "Hành vi vi phạm",
        "Hành vi bị cấm",
        "Hình thức xử phạt",
    },
}


@dataclass
class HybridIntent:
    anchor_terms: list[str]
    ll_keywords: list[str]
    hl_keywords: list[str]
    focus_buckets: list[str]


INTENT_PROMPT = """Return JSON with keys anchor_terms, low_level_keywords, high_level_keywords, focus_buckets.
Focus buckets must be chosen from: scope, conditions, responsibilities, violations.
Preserve Vietnamese legal wording.
"""


async def extract_hybrid_intent(query: str, history: list[dict[str, str]]) -> HybridIntent:
    raw = await gemini_chat_llm_func(
        prompt=query,
        system_prompt=INTENT_PROMPT,
        history=history,
        temperature=0.0,
    )
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    data = json.loads(match.group(0) if match else "{}")
    return HybridIntent(
        anchor_terms=data.get("anchor_terms", []),
        ll_keywords=data.get("low_level_keywords", []),
        hl_keywords=data.get("high_level_keywords", []),
        focus_buckets=data.get("focus_buckets", ["scope", "conditions", "responsibilities", "violations"]),
    )


def select_anchor_candidate(data: dict[str, Any], query: str, intent: HybridIntent) -> dict[str, Any] | None:
    best_candidate = None
    best_score = -1

    for entity in data.get("entities", []):
        if entity.get("entity_type") != "Điều khoản":
            continue

        score = 0
        lowered_name = (entity.get("entity_name") or "").lower()
        lowered_desc = (entity.get("description") or "").lower()

        for term in intent.anchor_terms + intent.ll_keywords:
            lowered_term = term.lower()
            if lowered_term and lowered_term in lowered_name:
                score += 5
            elif lowered_term and lowered_term in lowered_desc:
                score += 2

        for relationship in data.get("relationships", []):
            if relationship.get("src_id") == entity.get("entity_name"):
                score += 1

        if score > best_score:
            best_candidate = entity
            best_score = score

    return best_candidate


def bucket_expansion_chunks(data: dict[str, Any], anchor: dict[str, Any], intent: HybridIntent) -> dict[str, list[dict[str, Any]]]:
    buckets = {bucket: [] for bucket in intent.focus_buckets}
    anchor_name = anchor.get("entity_name")
    related_names_by_bucket = {bucket: set() for bucket in intent.focus_buckets}

    entity_type_lookup = {
        entity.get("entity_name"): entity.get("entity_type")
        for entity in data.get("entities", [])
    }

    for relationship in data.get("relationships", []):
        src = relationship.get("src_id")
        tgt = relationship.get("tgt_id")
        if anchor_name not in {src, tgt}:
            continue

        other_name = tgt if src == anchor_name else src
        other_type = entity_type_lookup.get(other_name)
        for bucket in intent.focus_buckets:
            if other_type in FOCUS_BUCKET_LABELS[bucket]:
                related_names_by_bucket[bucket].add((other_name or "").lower())

    for chunk in data.get("chunks", []):
        content = (chunk.get("content") or "").lower()
        for bucket in intent.focus_buckets:
            names = related_names_by_bucket[bucket]
            if any(name and name in content for name in names):
                buckets[bucket].append(chunk)

    return {key: value[: settings.HYBRID_BUCKET_CHUNK_LIMIT] for key, value in buckets.items() if value}


def build_hybrid_context(query: str, anchor: dict[str, Any], buckets: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        f"# Query\n{query}",
        "## Anchor Provision",
        json.dumps(anchor, ensure_ascii=False),
    ]
    if "scope" in buckets:
        lines.extend(["## Scope And Regulated Subjects", *[json.dumps(chunk, ensure_ascii=False) for chunk in buckets["scope"]]])
    if "conditions" in buckets:
        lines.extend(["## Conditions And Exceptions", *[json.dumps(chunk, ensure_ascii=False) for chunk in buckets["conditions"]]])
    if "responsibilities" in buckets:
        lines.extend(["## Responsibilities", *[json.dumps(chunk, ensure_ascii=False) for chunk in buckets["responsibilities"]]])
    if "violations" in buckets:
        lines.extend(["## Violations And Sanctions", *[json.dumps(chunk, ensure_ascii=False) for chunk in buckets["violations"]]])
    lines.append("## References")
    lines.append(f"- [{anchor.get('reference_id')}] {anchor.get('file_path')}")
    return "\n\n".join(lines)


def build_hybrid_system_prompt(context: str) -> str:
    return (
        "Answer in Vietnamese using only the provided traffic-law context.\n"
        "Start from the anchor provision, then expand only with grounded scope, condition, responsibility, violation, or sanction details.\n"
        "If the context is insufficient, say so explicitly.\n\n"
        f"{context}"
    )


async def run_hybrid_query(rag, query: str, history: list[dict[str, str]], *, stream: bool = False) -> str | AsyncIterator[str]:
    intent = await extract_hybrid_intent(query, history)
    retrieval = await rag.aquery_data(
        query,
        param=QueryParam(
            mode="hybrid",
            ll_keywords=intent.ll_keywords,
            hl_keywords=intent.hl_keywords,
            conversation_history=history,
            top_k=settings.HYBRID_TOP_K,
            chunk_top_k=settings.HYBRID_CHUNK_TOP_K,
        ),
    )
    data = retrieval.get("data", {})
    anchor = select_anchor_candidate(data, query, intent)
    if anchor is None:
        return "Tôi không có đủ thông tin được liên kết rõ ràng trong ngữ cảnh để trả lời câu hỏi này."
    buckets = bucket_expansion_chunks(data, anchor, intent)
    context = build_hybrid_context(query, anchor, buckets)
    return await gemini_chat_llm_func(
        prompt=query,
        system_prompt=build_hybrid_system_prompt(context),
        history=history,
        stream=stream,
        temperature=0.2,
    )
```

- [ ] **Step 4: Re-run the hybrid strategy tests**

Run:

```bash
pytest backend/tests/test_hybrid_query.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/core/hybrid_query.py backend/tests/test_hybrid_query.py
git commit -m "feat: add anchor-first hybrid query strategy"
```

---

### Task 4: Integrate The Hybrid Strategy Into The API And Frontend

**Files:**
- Modify: `backend/api/routes.py`
- Modify: `backend/tests/test_chat_route.py`
- Modify: `frontend/src/components/ChatInterface.tsx`

- [ ] **Step 1: Extend the route tests to cover hybrid orchestration and streaming**

Append to `backend/tests/test_chat_route.py`:

```python
def test_chat_comparison_uses_custom_hybrid_runner(monkeypatch):
    import backend.api.routes as routes

    fake_rag = FakeRAG()
    hybrid_calls = []

    async def fake_run_hybrid_query(rag, query, history, *, stream=False):
        hybrid_calls.append(
            {
                "rag": rag,
                "query": query,
                "history": history,
                "stream": stream,
            }
        )
        return "hybrid:anchor-first"

    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    request = ChatRequest(
        message="Ai chịu trách nhiệm và có ngoại lệ gì trong dịch vụ cứu hộ giao thông đường bộ?",
        history=[{"role": "user", "content": "Điều 79 nói về gì?"}],
        comparison_mode=True,
        stream=False,
    )

    response = asyncio.run(routes.chat(request))

    assert response.naive.response == "naive:ok"
    assert response.hybrid.response == "hybrid:anchor-first"
    assert hybrid_calls == [
        {
            "rag": fake_rag,
            "query": request.message,
            "history": request.history,
            "stream": False,
        }
    ]


def test_chat_stream_uses_hybrid_runner_chunks(monkeypatch):
    import backend.api.routes as routes

    fake_rag = FakeRAG()

    async def fake_stream():
        yield "phan 1"
        yield "phan 2"

    async def fake_run_hybrid_query(rag, query, history, *, stream=False):
        assert stream is True
        return fake_stream()

    monkeypatch.setattr(routes, "get_rag_engine", lambda: fake_rag)
    monkeypatch.setattr(routes, "run_hybrid_query", fake_run_hybrid_query)

    request = ChatRequest(
        message="Ai chịu trách nhiệm cứu hộ giao thông đường bộ?",
        history=[],
        comparison_mode=False,
        stream=True,
    )

    response = asyncio.run(routes.chat(request))

    async def collect(body_iterator):
        parts = []
        async for item in body_iterator:
            parts.append(item.decode() if isinstance(item, bytes) else item)
        return "".join(parts)

    payload = asyncio.run(collect(response.body_iterator))

    assert '"mode": "hybrid"' in payload
    assert "phan 1" in payload
    assert "phan 2" in payload
```

- [ ] **Step 2: Run the focused integration tests to verify they fail**

Run:

```bash
pytest backend/tests/test_chat_route.py -v
```

Expected: FAIL because `backend/api/routes.py` does not yet import or call `run_hybrid_query(...)`.

- [ ] **Step 3: Integrate the hybrid service and send chat history from the frontend**

Update `backend/api/routes.py` imports:

```python
from backend.core.hybrid_query import run_hybrid_query
```

Replace the non-streaming `hybrid` path:

```python
if request.comparison_mode:
    naive_response = await rag.aquery(
        request.message,
        param=_build_query_param("naive", request),
    )
    hybrid_response = await run_hybrid_query(
        rag,
        request.message,
        request.history,
        stream=False,
    )
    return ComparisonResponse(
        naive=ChatResponse(response=naive_response, mode="naive"),
        hybrid=ChatResponse(response=hybrid_response, mode="hybrid"),
    )
else:
    response = await run_hybrid_query(
        rag,
        request.message,
        request.history,
        stream=False,
    )
    return ChatResponse(response=response, mode="hybrid")
```

Replace the streaming `hybrid` path:

```python
t2 = asyncio.create_task(
    stream_wrapper(
        run_hybrid_query(
            rag,
            request.message,
            request.history,
            stream=True,
        ),
        "hybrid",
    )
)
```

and:

```python
generator = await run_hybrid_query(
    rag,
    request.message,
    request.history,
    stream=True,
)
```

Update `frontend/src/components/ChatInterface.tsx` to send recent history. Add a helper near the top of the component:

```ts
  const buildHistoryPayload = (items: Message[]) => {
    return items
      .filter((msg) => msg.id !== 'initial')
      .flatMap((msg) => {
        if (msg.comparison) {
          const hybridContent = msg.comparison.hybrid.content.trim()
          return hybridContent ? [{ role: 'assistant', content: hybridContent }] : []
        }

        const content = (msg.content || '').trim()
        return content ? [{ role: msg.role, content }] : []
      })
      .slice(-8)
  }
```

Then include it in the request body:

```ts
      const historyPayload = buildHistoryPayload(messages)

      const response = await fetch(`${(import.meta as any).env.VITE_API_BASE_URL || 'http://localhost:8000/api'}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          history: historyPayload,
          comparison_mode: comparisonMode,
          stream: true,
        })
      })
```

This deliberately persists the completed `hybrid` answer as the assistant-side history turn when comparison mode is on, which keeps follow-up behavior aligned with the primary synthesis mode.

- [ ] **Step 4: Re-run backend tests and the frontend build**

Run:

```bash
pytest backend/tests/test_chat_route.py backend/tests/test_hybrid_query.py -v
cd frontend && npm run build
```

Expected:

- pytest: PASS
- Vite build: PASS with generated production bundle output

- [ ] **Step 5: Commit**

```bash
git add backend/api/routes.py backend/tests/test_chat_route.py frontend/src/components/ChatInterface.tsx
git commit -m "feat: wire anchor-first hybrid into chat flow"
```

---

### Task 5: Add Benchmark Docs, Re-Index Notes, And Final Project Documentation

**Files:**
- Create: `docs/hybrid-benchmark.md`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PROJECT_STRUCTURE.md`

- [ ] **Step 1: Write the benchmark and re-index documentation**

Create `docs/hybrid-benchmark.md`:

```markdown
# Hybrid Traffic-Law Benchmark

## Purpose

This benchmark measures whether `hybrid` outperforms `naive` on synthesis-oriented traffic-law questions.

## Query Groups

1. Scope of application and regulated subjects
2. Conditions and exceptions
3. Responsibilities of subjects or authorities
4. Violations and related sanctions
5. One anchor provision with supporting provisions across nearby articles

## Example Cases

- Query: `Đối tượng nào được phép hoặc không được phép sử dụng xe cứu hộ giao thông đường bộ?`
  - Expected anchor: `Điều 79`
  - Expected focus: `scope`, `conditions`

- Query: `Ai chịu trách nhiệm quản lý nhà nước về hoạt động đường bộ và trách nhiệm đó được phân chia như thế nào?`
  - Expected anchor: `Điều 82`
  - Expected focus: `responsibilities`, `scope`

- Query: `Những hành vi nào bị cấm và có thể dẫn đến xử phạt trong hoạt động đường bộ?`
  - Expected anchor: article varies by corpus coverage
  - Expected focus: `violations`

## Re-Index Requirement

Changing `ENTITY_TYPES` requires rebuilding the graph.

Minimum validation checklist after deployment:

1. Start backend with the updated ontology
2. Use a fresh or cleared LightRAG database state
3. Re-upload `35-2024-qh15.pdf` and `36-2024-qh15.pdf`
4. Re-run the benchmark queries in comparison mode
5. Confirm `hybrid` shows one legal anchor and grounded supporting points
```

- [ ] **Step 2: Update `.env.example` and the main docs**

Append the new knobs to `.env.example`:

```dotenv
HYBRID_MAX_HISTORY_MESSAGES=8
HYBRID_TOP_K=20
HYBRID_CHUNK_TOP_K=12
HYBRID_ANCHOR_CHUNK_LIMIT=3
HYBRID_BUCKET_CHUNK_LIMIT=2
```

Update `README.md` with a short architecture note:

```markdown
## Hybrid Retrieval Mode

The comparison UI remains `naive vs hybrid`.

- `naive` is the exact-passage baseline
- `hybrid` is tuned for traffic-law synthesis questions and now follows an anchor-first strategy:
  1. identify one central `Điều khoản`
  2. expand only into grounded scope, conditions, responsibilities, violations, and sanctions

Changing `ENTITY_TYPES` requires re-indexing the uploaded legal corpus.
```

Update `ARCHITECTURE.md` with the new backend query flow:

```markdown
### Hybrid Query Flow

1. frontend sends message + recent conversation history
2. backend extracts traffic-law intent for `hybrid`
3. LightRAG `aquery_data(...)` retrieves structured KG data
4. backend selects one primary `Điều khoản` anchor
5. backend groups supporting chunks by legal function
6. Gemini generates the final grounded answer from the anchor-first context
```

Update `PROJECT_STRUCTURE.md` with the new module:

```markdown
- `backend/core/hybrid_query.py`: anchor-first hybrid retrieval orchestration for traffic-law synthesis queries
- `docs/hybrid-benchmark.md`: manual synthesis benchmark and re-index checklist
```

- [ ] **Step 3: Run the final verification commands**

Run:

```bash
pytest backend/tests/test_chat_route.py backend/tests/test_hybrid_query.py backend/tests/test_llm_services.py backend/tests/test_rag_engine.py backend/tests/test_upload_route.py -v
cd frontend && npm run build
```

Expected:

- all listed backend tests PASS
- frontend production build PASS

- [ ] **Step 4: Commit**

```bash
git add docs/hybrid-benchmark.md .env.example README.md ARCHITECTURE.md PROJECT_STRUCTURE.md
git commit -m "docs: add hybrid benchmark and reindex guidance"
```
