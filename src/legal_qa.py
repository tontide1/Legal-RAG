from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pipeline_utils import get_configured_gemini_model

DEFAULT_ABSTAIN_ANSWER = "Tôi không tìm thấy căn cứ pháp lý đủ rõ cho trường hợp này."
DEFAULT_GENERATION_ERROR_ANSWER = (
    "Tôi không thể tạo câu trả lời lúc này. Vui lòng thử lại sau hoặc đổi API key."
)
DEFAULT_NER_BACKEND = "bilstm"

SERIALIZED_NODE_KEYS = (
    "node_id",
    "label",
    "name",
    "value",
    "bm25",
    "cosine",
    "graph_sum",
    "combined_score",
    "final_score",
)


def build_answer_chain():
    from dotenv import load_dotenv
    from langchain_core.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file")

    gemini_model = get_configured_gemini_model()
    answer_prompt = PromptTemplate.from_template(
        """Hãy trở thành chuyên gia tư vấn luật tại Việt Nam.
Câu hỏi của người dùng: {query}
Trả lời dựa vào thông tin sau:
{source_information}
Yêu cầu:
1. Trả lời ngắn gọn, rõ ràng
2. Nếu không có thông tin, trả lời "Tôi không tìm thấy căn cứ pháp lý cho trường hợp này" và gợi ý cách tìm kiếm ở nguồn khác
3. Kèm điều luật liên quan"""
    )
    answer_model = ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=google_api_key,
        temperature=0.3,
        max_output_tokens=500,
    )
    return answer_prompt | answer_model


def serialize_retrieved_nodes(nodes: list[dict]) -> list[dict]:
    serialized_nodes = []
    for node in nodes:
        serialized_node = {}
        for key in SERIALIZED_NODE_KEYS:
            if key in node:
                serialized_node[key] = node[key]
        serialized_nodes.append(serialized_node)
    return serialized_nodes


def extract_citations_from_nodes(nodes: list[dict]) -> list[dict]:
    citations = []
    for node in nodes:
        citations.append(
            {
                "node_id": node.get("node_id"),
                "label": node.get("label"),
                "name": node.get("name"),
                "score": node.get("final_score", node.get("combined_score")),
            }
        )
    return citations


def build_context_text(nodes: list[dict]) -> str:
    lines = []
    for node in nodes:
        label = node.get("label", "Unknown")
        name = node.get("name", "")
        value = node.get("value", "")
        lines.append(f"- {label} ({name}): {value}")
    return "\n".join(lines)


def resolve_ner_backend() -> str:
    backend = os.getenv("NER_BACKEND", DEFAULT_NER_BACKEND).strip().lower()
    if backend in {"bilstm", "phobert"}:
        return backend
    return DEFAULT_NER_BACKEND


def _default_ner_infer(query: str) -> list[str]:
    backend = resolve_ner_backend()

    if backend == "phobert":
        from NER import phobert_ner

        checkpoint_dir = os.getenv(
            "PHOBERT_NER_CHECKPOINT",
            str(CODE_ROOT / "NER" / "checkpoints" / "phobert_article_ner"),
        )
        max_length = int(os.getenv("PHOBERT_NER_MAX_LENGTH", "128"))
        _, _, ner_entities = phobert_ner.infer(
            query,
            checkpoint_dir=checkpoint_dir,
            max_length=max_length,
        )
        return ner_entities

    from NER import ner

    _, _, ner_entities = ner.infer(
        query,
        model_path=str(CODE_ROOT / "NER" / "bilstm_ner.pt"),
    )
    return ner_entities


def _default_retrieve(query: str, ner_entities: list[str]) -> list[dict]:
    from retrive import multi_retr

    return multi_retr.retrieve_entity(query, ner_entities if ner_entities else None, verbose=False)


def _default_generate_answer(query: str, context_text: str) -> str:
    response = build_answer_chain().invoke({"query": query, "source_information": context_text})
    return response.content if hasattr(response, "content") else str(response)


def run_legal_qa(
    query: str,
    *,
    ner_infer_fn: Callable[[str], list[str]] | None = None,
    retrieve_fn: Callable[[str, list[str]], list[dict]] | None = None,
    generate_answer_fn: Callable[[str, str], str] | None = None,
    min_final_score: float | None = None,
) -> dict:
    started_at = time.perf_counter()
    cleaned_query = query.strip()
    errors: list[str] = []

    result = {
        "query": cleaned_query,
        "ner_entities": [],
        "retrieved_nodes": [],
        "context_text": "",
        "answer_text": DEFAULT_ABSTAIN_ANSWER,
        "citations": [],
        "scores": {},
        "timings": {},
        "errors": errors,
    }

    if not cleaned_query:
        errors.append("Query không được để trống.")
        result["timings"]["total_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        return result

    ner_infer = ner_infer_fn or _default_ner_infer
    retrieve = retrieve_fn or _default_retrieve
    generate_answer = generate_answer_fn or _default_generate_answer

    ner_started_at = time.perf_counter()
    ner_entities = ner_infer(cleaned_query)
    result["ner_entities"] = list(ner_entities)
    result["timings"]["ner_ms"] = round((time.perf_counter() - ner_started_at) * 1000, 3)

    retrieval_started_at = time.perf_counter()
    retrieved_nodes = retrieve(cleaned_query, result["ner_entities"])
    serialized_nodes = serialize_retrieved_nodes(retrieved_nodes)
    result["retrieved_nodes"] = serialized_nodes
    result["citations"] = extract_citations_from_nodes(serialized_nodes)
    result["context_text"] = build_context_text(serialized_nodes)
    result["timings"]["retrieval_ms"] = round((time.perf_counter() - retrieval_started_at) * 1000, 3)

    top_score = None
    if serialized_nodes:
        top_score = serialized_nodes[0].get("final_score", serialized_nodes[0].get("combined_score"))
    result["scores"] = {
        "retrieved_count": len(serialized_nodes),
        "top_score": top_score,
    }

    should_abstain = not serialized_nodes
    if min_final_score is not None and top_score is not None and top_score < min_final_score:
        should_abstain = True

    if not should_abstain:
        generation_started_at = time.perf_counter()
        try:
            result["answer_text"] = generate_answer(cleaned_query, result["context_text"])
        except Exception as exc:
            error_text = str(exc)
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                errors.append("Lỗi Gemini: vượt quota/rate limit (429 RESOURCE_EXHAUSTED).")
            else:
                errors.append(f"Lỗi khi gọi Gemini: {exc}")
            result["answer_text"] = DEFAULT_GENERATION_ERROR_ANSWER
        result["timings"]["generation_ms"] = round((time.perf_counter() - generation_started_at) * 1000, 3)
    else:
        result["timings"]["generation_ms"] = 0.0

    result["timings"]["total_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
    return result


__all__ = [
    "DEFAULT_ABSTAIN_ANSWER",
    "DEFAULT_GENERATION_ERROR_ANSWER",
    "DEFAULT_NER_BACKEND",
    "build_context_text",
    "extract_citations_from_nodes",
    "resolve_ner_backend",
    "run_legal_qa",
    "serialize_retrieved_nodes",
]
