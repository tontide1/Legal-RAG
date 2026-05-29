import asyncio
import importlib
import sys
import types


def _install_fake_lightrag(monkeypatch):
    lightrag_module = types.ModuleType("lightrag")

    class FakeQueryParam:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    lightrag_module.QueryParam = FakeQueryParam
    monkeypatch.setitem(sys.modules, "lightrag", lightrag_module)


def _load_module(monkeypatch):
    _install_fake_lightrag(monkeypatch)
    sys.modules.pop("backend.core.hybrid_query", None)
    return importlib.import_module("backend.core.hybrid_query")


def test_select_anchor_candidate_prefers_matching_dieu_khoan(monkeypatch):
    hybrid_query = _load_module(monkeypatch)

    intent = hybrid_query.HybridIntent(
        anchor_terms=["Điều 9", "điều 9"],
        ll_keywords=["điều 9"],
        hl_keywords=["quy tắc giao thông"],
        focus_buckets=["scope", "conditions"],
    )
    data = {
        "entities": [
            {"entity_name": "Điều 12", "entity_type": "Điều khoản", "description": "Quy định khác"},
            {"entity_name": "Điều 9", "entity_type": "Điều khoản", "description": "Quy tắc chung"},
            {"entity_name": "Điều 9", "entity_type": "Khái niệm pháp lý", "description": "Không phải anchor ưu tiên"},
        ],
        "relationships": [],
        "chunks": [
            {
                "id": "c-anchor",
                "content": "Điều 9 Luật Trật tự, an toàn giao thông đường bộ quy định quy tắc chung.",
                "metadata": {"source": "law.pdf", "article": "Điều 9"},
            }
        ],
        "references": [
            {"chunk_id": "c-anchor", "source": "law.pdf", "article": "Điều 9"}
        ],
    }

    anchor = hybrid_query.select_anchor_candidate(data, "Điều 9 quy định gì?", intent)

    assert anchor is not None
    assert anchor["entity_name"] == "Điều 9"
    assert anchor["entity_type"] == "Điều khoản"


def test_bucket_expansion_groups_scope_and_conditions(monkeypatch):
    hybrid_query = _load_module(monkeypatch)

    intent = hybrid_query.HybridIntent(
        anchor_terms=["Điều 9"],
        ll_keywords=["áp dụng", "trường hợp"],
        hl_keywords=["quy tắc giao thông"],
        focus_buckets=["scope", "conditions"],
    )
    anchor = {"entity_name": "Điều 9", "entity_type": "Điều khoản"}
    data = {
        "entities": [],
        "relationships": [],
        "chunks": [
            {
                "id": "scope-1",
                "content": "Điều 9 áp dụng đối với người điều khiển xe cơ giới khi tham gia giao thông đường bộ.",
                "metadata": {"source": "law.pdf", "article": "Điều 9"},
            },
            {
                "id": "cond-1",
                "content": "Trong trường hợp chuyển hướng, người lái xe phải giảm tốc độ và bật tín hiệu báo hướng rẽ.",
                "metadata": {"source": "law.pdf", "article": "Điều 9"},
            },
            {
                "id": "misc-1",
                "content": "Người vi phạm có thể bị xử phạt theo nghị định liên quan.",
                "metadata": {"source": "decree.pdf", "article": "Điều 4"},
            },
        ],
        "references": [
            {"chunk_id": "scope-1", "source": "law.pdf", "article": "Điều 9"},
            {"chunk_id": "cond-1", "source": "law.pdf", "article": "Điều 9"},
            {"chunk_id": "misc-1", "source": "decree.pdf", "article": "Điều 4"},
        ],
    }

    buckets = hybrid_query.bucket_expansion_chunks(data, anchor, intent)

    assert [chunk["id"] for chunk in buckets["scope"]] == ["scope-1"]
    assert [chunk["id"] for chunk in buckets["conditions"]] == ["cond-1"]
    assert buckets["responsibilities"] == []
    assert buckets["violations"] == []


def test_build_hybrid_context_orders_sections_and_includes_references(monkeypatch):
    hybrid_query = _load_module(monkeypatch)

    anchor = {
        "entity_name": "Điều 9",
        "entity_type": "Điều khoản",
        "description": "Quy tắc chung",
        "anchor_chunks": [
            {
                "id": "anchor-1",
                "content": "Điều 9 quy định người tham gia giao thông phải chấp hành hệ thống báo hiệu đường bộ.",
                "reference_label": "law.pdf - Điều 9",
            }
        ],
    }
    buckets = {
        "scope": [
            {
                "id": "scope-1",
                "content": "Áp dụng đối với người điều khiển phương tiện và người đi bộ.",
                "reference_label": "law.pdf - Điều 9",
            }
        ],
        "conditions": [
            {
                "id": "cond-1",
                "content": "Khi chuyển hướng phải quan sát và giảm tốc độ.",
                "reference_label": "law.pdf - Điều 9",
            }
        ],
        "responsibilities": [],
        "violations": [
            {
                "id": "viol-1",
                "content": "Không chấp hành hiệu lệnh đèn tín hiệu là hành vi vi phạm.",
                "reference_label": "decree.pdf - Điều 4",
            }
        ],
    }

    context = hybrid_query.build_hybrid_context("Điều 9 quy định gì?", anchor, buckets)

    sections = [
        "## Cau hoi",
        "## Dieu khoan trung tam",
        "## Pham vi ap dung",
        "## Dieu kien va tinh huong",
        "## Hanh vi vi pham",
    ]
    positions = [context.index(section) for section in sections]
    assert positions == sorted(positions)
    assert "Nguon: law.pdf - Điều 9" in context
    assert "Nguon: decree.pdf - Điều 4" in context


def test_run_hybrid_query_returns_insufficient_context_when_no_anchor(monkeypatch):
    hybrid_query = _load_module(monkeypatch)

    monkeypatch.setattr(
        hybrid_query,
        "extract_hybrid_intent",
        lambda query, history: hybrid_query.HybridIntent(
            anchor_terms=["Điều 99"],
            ll_keywords=["điều 99"],
            hl_keywords=["quy định"],
            focus_buckets=["scope"],
        ),
    )

    class FakeRAG:
        def __init__(self):
            self.calls = []

        async def aquery_data(self, query, param):
            self.calls.append((query, param))
            return {
                "data": {
                    "entities": [
                        {"entity_name": "Điều 12", "entity_type": "Điều khoản", "description": "Khác"}
                    ],
                    "relationships": [],
                    "chunks": [],
                    "references": [],
                }
            }

    rag = FakeRAG()
    history = [
        {"role": "user", "content": "Tin nhắn 1"},
        {"role": "assistant", "content": "Tin nhắn 2"},
        {"role": "user", "content": "Tin nhắn 3"},
    ]

    response = asyncio.run(
        hybrid_query.run_hybrid_query(rag, "Điều 99 quy định gì?", history)
    )

    assert "không đủ ngữ cảnh" in response.lower()
    assert len(rag.calls) == 1
    _, param = rag.calls[0]
    assert param.mode == "hybrid"
    assert param.include_references is True
    assert param.conversation_history == history
