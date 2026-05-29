from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from backend.config import settings


INSUFFICIENT_CONTEXT_MESSAGE = (
    "Không đủ ngữ cảnh để xác định điều khoản trung tâm cho câu hỏi này. "
    "Vui lòng nêu rõ điều, khoản hoặc tình huống giao thông cần tra cứu."
)

_ARTICLE_PATTERN = re.compile(r"(điều\s+\d+[a-zA-Z]*)", re.IGNORECASE)
_CLAUSE_PATTERN = re.compile(r"(khoản\s+\d+[a-zA-Z]*)", re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

_BUCKET_RULES: dict[str, tuple[str, ...]] = {
    "scope": (
        "áp dụng",
        "đối với",
        "phạm vi",
        "người điều khiển",
        "người tham gia",
        "phương tiện",
    ),
    "conditions": (
        "trường hợp",
        "khi",
        "nếu",
        "điều kiện",
        "chuyển hướng",
        "giảm tốc độ",
    ),
    "responsibilities": (
        "phải",
        "có trách nhiệm",
        "chịu trách nhiệm",
        "nghĩa vụ",
        "chấp hành",
    ),
    "violations": (
        "vi phạm",
        "bị cấm",
        "không được",
        "xử phạt",
        "hành vi",
    ),
}

_BUCKET_TITLES = {
    "scope": "## Pham vi ap dung",
    "conditions": "## Dieu kien va tinh huong",
    "responsibilities": "## Trach nhiem va nghia vu",
    "violations": "## Hanh vi vi pham",
}


@dataclass(slots=True)
class HybridIntent:
    anchor_terms: list[str]
    ll_keywords: list[str]
    hl_keywords: list[str]
    focus_buckets: list[str]


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_PATTERN.sub(" ", text)


def _normalize_for_match(value: Any) -> str:
    return _normalize_text(value).casefold()


def _term_in_text(text: Any, term: Any) -> bool:
    normalized_text = _normalize_for_match(text)
    normalized_term = _normalize_for_match(term)
    if not normalized_text or not normalized_term:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_term)}(?!\w)"
    return bool(re.search(pattern, normalized_text))


def _trim_history(history: list[dict] | None) -> list[dict]:
    if not history:
        return []
    limit = max(int(getattr(settings, "HYBRID_MAX_HISTORY_MESSAGES", 0) or 0), 0)
    if limit == 0:
        return []
    return list(history[-limit:])


def _extract_anchor_terms(text: str) -> list[str]:
    matches = _ARTICLE_PATTERN.findall(text) + _CLAUSE_PATTERN.findall(text)
    terms: list[str] = []
    seen: set[str] = set()
    for match in matches:
        normalized = _normalize_text(match)
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(normalized)
    return terms


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.casefold())
    stop_words = {
        "la",
        "là",
        "ve",
        "về",
        "cho",
        "toi",
        "tôi",
        "quy",
        "dinh",
        "định",
        "gi",
        "gì",
        "nao",
        "nào",
        "the",
        "thế",
        "mot",
        "một",
        "cua",
        "của",
        "trong",
    }
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 3 or token in stop_words or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _derive_focus_buckets(text: str) -> list[str]:
    focus_buckets: list[str] = []

    if any(_term_in_text(text, term) for term in ("áp dụng", "đối với", "phạm vi")):
        focus_buckets.append("scope")
    if any(_term_in_text(text, term) for term in ("điều kiện", "trường hợp", "khi", "nếu")):
        focus_buckets.append("conditions")
    if any(_term_in_text(text, term) for term in ("trách nhiệm", "nghĩa vụ", "phải")):
        focus_buckets.append("responsibilities")
    if any(_term_in_text(text, term) for term in ("vi phạm", "xử phạt", "bị cấm")):
        focus_buckets.append("violations")

    if not focus_buckets:
        focus_buckets = ["scope", "conditions", "responsibilities", "violations"]
    return focus_buckets


def extract_hybrid_intent(query: str, history: list[dict] | None) -> HybridIntent:
    trimmed_history = _trim_history(history)
    history_text = " ".join(_normalize_text(item.get("content", "")) for item in trimmed_history)
    combined_text = _normalize_text(f"{history_text} {query}")

    anchor_terms = _extract_anchor_terms(combined_text)
    ll_keywords = _extract_keywords(query)
    hl_keywords = _extract_keywords(combined_text)
    focus_buckets = _derive_focus_buckets(query)

    return HybridIntent(
        anchor_terms=anchor_terms,
        ll_keywords=ll_keywords[:8],
        hl_keywords=hl_keywords[:8],
        focus_buckets=focus_buckets,
    )


def _reference_map(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    references = data.get("references") or []
    result: dict[str, dict[str, Any]] = {}
    for reference in references:
        chunk_id = str(reference.get("chunk_id") or "").strip()
        if chunk_id:
            result[chunk_id] = reference
    return result


def _build_reference_label(chunk: dict[str, Any], reference: dict[str, Any] | None) -> str:
    metadata = chunk.get("metadata") or {}
    source = (
        (reference or {}).get("source")
        or metadata.get("source")
        or metadata.get("document")
        or "unknown"
    )
    article = (
        (reference or {}).get("article")
        or metadata.get("article")
        or metadata.get("label")
        or ""
    )
    if article:
        return f"{source} - {article}"
    return str(source)


def _enrich_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    ref_map = _reference_map(data)
    enriched: list[dict[str, Any]] = []
    for raw_chunk in data.get("chunks") or []:
        chunk = dict(raw_chunk)
        chunk_id = str(chunk.get("id") or "").strip()
        reference = ref_map.get(chunk_id)
        chunk["reference"] = reference
        chunk["reference_label"] = _build_reference_label(chunk, reference)
        enriched.append(chunk)
    return enriched


def _chunk_mentions_anchor(chunk: dict[str, Any], anchor_name: str) -> bool:
    anchor_key = _normalize_for_match(anchor_name)
    content = _normalize_for_match(chunk.get("content", ""))
    metadata = chunk.get("metadata") or {}
    article = _normalize_for_match(metadata.get("article", ""))
    reference = chunk.get("reference") or {}
    reference_article = _normalize_for_match(reference.get("article", ""))
    return bool(anchor_key and (anchor_key in content or anchor_key == article or anchor_key == reference_article))


def _anchor_chunks(data: dict[str, Any], anchor_name: str) -> list[dict[str, Any]]:
    chunks = [chunk for chunk in _enrich_chunks(data) if _chunk_mentions_anchor(chunk, anchor_name)]
    limit = max(int(getattr(settings, "HYBRID_ANCHOR_CHUNK_LIMIT", 0) or 0), 0)
    if limit:
        return chunks[:limit]
    return chunks


def _score_anchor_candidate(entity: dict[str, Any], query: str, intent: HybridIntent) -> int:
    score = 0
    entity_type = _normalize_for_match(entity.get("entity_type", ""))
    entity_name = _normalize_for_match(entity.get("entity_name", ""))
    description = _normalize_for_match(entity.get("description", ""))
    query_key = _normalize_for_match(query)
    matched_anchor_term = False

    if entity_type == "điều khoản":
        score += 100

    for term in intent.anchor_terms:
        term_key = _normalize_for_match(term)
        if term_key and term_key == entity_name:
            score += 200
            matched_anchor_term = True
        elif term_key and term_key in entity_name:
            score += 120
            matched_anchor_term = True
        elif term_key and term_key in description:
            score += 60
            matched_anchor_term = True

    if intent.anchor_terms and not matched_anchor_term:
        return 0

    if entity_name and entity_name in query_key:
        score += 40

    for keyword in intent.ll_keywords:
        if keyword and keyword in description:
            score += 5

    return score


def select_anchor_candidate(data: dict[str, Any], query: str, intent: HybridIntent) -> dict[str, Any] | None:
    candidates = list(data.get("entities") or [])
    if candidates:
        ranked = sorted(
            candidates,
            key=lambda entity: _score_anchor_candidate(entity, query, intent),
            reverse=True,
        )
        best = ranked[0] if ranked else None
        best_score = _score_anchor_candidate(best, query, intent) if best is not None else 0
        if best is not None and best_score > 0:
            anchor = dict(best)
            anchor["anchor_chunks"] = _anchor_chunks(data, anchor.get("entity_name", ""))
            return anchor

    synthetic_terms = intent.anchor_terms or _extract_anchor_terms(query)
    for anchor_name in synthetic_terms:
        anchor_chunks = _anchor_chunks(data, anchor_name)
        if anchor_chunks:
            return {
                "entity_name": anchor_name,
                "entity_type": "Điều khoản",
                "description": "",
                "anchor_chunks": anchor_chunks,
            }

    return None


def _chunk_bucket(chunk: dict[str, Any], intent: HybridIntent) -> str | None:
    content = _normalize_for_match(chunk.get("content", ""))
    matched_bucket: str | None = None
    matched_score = -1

    for bucket, terms in _BUCKET_RULES.items():
        term_score = sum(1 for term in terms if _term_in_text(content, term))
        if term_score <= 0:
            continue

        score = term_score
        if bucket in intent.focus_buckets:
            score += 2
        if score > matched_score:
            matched_bucket = bucket
            matched_score = score

    if matched_score <= 0:
        return None
    return matched_bucket


def bucket_expansion_chunks(
    data: dict[str, Any],
    anchor: dict[str, Any],
    intent: HybridIntent,
) -> dict[str, list[dict[str, Any]]]:
    bucketed = {
        "scope": [],
        "conditions": [],
        "responsibilities": [],
        "violations": [],
    }
    seen_ids = {
        str(chunk.get("id") or "")
        for chunk in anchor.get("anchor_chunks", [])
        if chunk.get("id")
    }
    limit = max(int(getattr(settings, "HYBRID_BUCKET_CHUNK_LIMIT", 0) or 0), 0)

    for chunk in _enrich_chunks(data):
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id or chunk_id in seen_ids:
            continue
        if not _chunk_mentions_anchor(chunk, anchor.get("entity_name", "")):
            continue

        bucket = _chunk_bucket(chunk, intent)
        if bucket is None:
            continue
        if limit and len(bucketed[bucket]) >= limit:
            continue

        bucketed[bucket].append(chunk)
        seen_ids.add(chunk_id)

    return bucketed


def _format_chunk_lines(chunks: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for chunk in chunks:
        content = _normalize_text(chunk.get("content", ""))
        if not content:
            continue
        lines.append(f"- {content}")
        reference_label = _normalize_text(chunk.get("reference_label", ""))
        if reference_label:
            lines.append(f"  Nguon: {reference_label}")
    return lines


def build_hybrid_context(
    query: str,
    anchor: dict[str, Any],
    buckets: dict[str, list[dict[str, Any]]],
) -> str:
    sections = [
        "## Cau hoi",
        _normalize_text(query),
        "",
        "## Dieu khoan trung tam",
        f"- {anchor.get('entity_name', '').strip()} ({anchor.get('entity_type', '').strip()})".strip(),
    ]

    description = _normalize_text(anchor.get("description", ""))
    if description:
        sections.append(f"  Mo ta: {description}")

    anchor_lines = _format_chunk_lines(anchor.get("anchor_chunks", []))
    if anchor_lines:
        sections.extend(anchor_lines)

    ordered_buckets = ["scope", "conditions", "responsibilities", "violations"]
    for bucket_name in ordered_buckets:
        chunks = buckets.get(bucket_name) or []
        if not chunks:
            continue
        sections.extend(["", _BUCKET_TITLES[bucket_name]])
        sections.extend(_format_chunk_lines(chunks))

    return "\n".join(sections).strip()


def build_hybrid_system_prompt(context: str) -> str:
    return (
        "Bạn là trợ lý pháp lý giao thông. "
        "Chỉ sử dụng ngữ cảnh đã cung cấp để trả lời bằng tiếng Việt. "
        "Trả lời trực tiếp, nêu rõ điều khoản trung tâm trước, sau đó giải thích theo từng nhóm thông tin phù hợp. "
        "Nếu dữ liệu chưa đủ để kết luận, hãy nói rõ giới hạn.\n\n"
        f"{context}"
    )


async def run_hybrid_query(
    rag,
    query: str,
    history: list[dict] | None,
    *,
    stream: bool = False,
):
    from lightrag import QueryParam

    trimmed_history = _trim_history(history)
    intent = extract_hybrid_intent(query, trimmed_history)
    top_k = int(getattr(settings, "HYBRID_TOP_K", 20))
    chunk_top_k = int(getattr(settings, "HYBRID_CHUNK_TOP_K", 12))

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
        ),
    )

    data = (result or {}).get("data") or {}
    anchor = select_anchor_candidate(data, query, intent)
    if anchor is None:
        return INSUFFICIENT_CONTEXT_MESSAGE

    buckets = bucket_expansion_chunks(data, anchor, intent)
    context = build_hybrid_context(query, anchor, buckets)
    system_prompt = build_hybrid_system_prompt(context)
    from backend.core.llm_services import gemini_chat_llm_func

    return await gemini_chat_llm_func(
        query,
        system_prompt=system_prompt,
        history=trimmed_history,
        stream=stream,
    )
