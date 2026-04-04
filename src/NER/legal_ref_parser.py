from __future__ import annotations

import re

_STOP_TOKENS = frozenset({
    "quy", "định", "là", "nói", "thế", "nào", "ra", "sao", "gì",
    "của", "trong", "theo", "về", "đối", "với", "và", "hoặc",
    "áp", "dụng", "cho", "có", "không", "được", "bị", "phải",
    "thì", "mà", "như", "từ", "đến", "khi", "nếu", "sau", "trước",
})

_PUNCTUATION = frozenset({".", ",", ";", ":", "?", "!", "(", ")", "[", "]", "{", "}"})

_DECREE_PATTERN = re.compile(
    r"(?i)\bnghị\s+định\s+(\d+/\d{4}/nđ-cp)\b"
)

_KNOWN_LAWS = frozenset({
    "hải quan",
    "doanh nghiệp",
    "lao động",
})

_KNOWN_CODE_LAWS = frozenset({
    "dân sự",
    "lao động",
})

_LAW_ANCHOR = "Luật"
_CODE_LAW_ANCHOR = "Bộ luật"


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_entity_text(text: str) -> str:
    cleaned = normalize_whitespace(text)
    parts = cleaned.split()
    if not parts:
        return ""
    result: list[str] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        if i == 0:
            result.append(part[0].upper() + part[1:] if len(part) > 1 else part.upper())
        else:
            result.append(part[0].upper() + part[1:] if len(part) > 1 else part.upper())
    return " ".join(result)


def _normalize_decree_code(code: str) -> str:
    parts = code.split("/")
    if len(parts) == 3:
        parts[2] = "NĐ-CP"
    return "/".join(parts)


def extract_decree_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in _DECREE_PATTERN.finditer(query):
        code = match.group(1)
        normalized_code = _normalize_decree_code(code)
        entity_text = f"Nghị định {normalized_code}"
        entities.append(entity_text)
    return entities


def _extract_phrase_entity(
    query: str,
    anchor: str,
    known_names: frozenset[str],
    max_tokens: int = 3,
) -> list[str]:
    entities: list[str] = []
    lower_query = query.lower()
    anchor_lower = anchor.lower()

    start = 0
    while True:
        pos = lower_query.find(anchor_lower, start)
        if pos == -1:
            break

        remaining = query[pos + len(anchor):]
        remaining_tokens = remaining.split()

        name_tokens: list[str] = []
        for token in remaining_tokens:
            token_stripped = token.strip(".,;:?!()[]{}\"'")
            token_lower = token_stripped.lower()
            if not token_stripped:
                break
            if token_lower in _PUNCTUATION:
                break
            if token_lower in _STOP_TOKENS:
                break
            name_tokens.append(token_stripped)
            if len(name_tokens) >= max_tokens:
                break

        if name_tokens:
            candidate_name = " ".join(name_tokens).lower()
            if candidate_name in known_names:
                full_entity = f"{anchor} {' '.join(name_tokens)}"
                entities.append(normalize_entity_text(full_entity))

        start = pos + len(anchor)

    return entities


def extract_law_entities(query: str) -> list[str]:
    return _extract_phrase_entity(query, _LAW_ANCHOR, _KNOWN_LAWS)


def extract_code_law_entities(query: str) -> list[str]:
    return _extract_phrase_entity(query, _CODE_LAW_ANCHOR, _KNOWN_CODE_LAWS)


def extract_legal_document_entities(query: str) -> list[str]:
    all_entities: list[str] = []
    all_entities.extend(extract_decree_entities(query))
    all_entities.extend(extract_law_entities(query))
    all_entities.extend(extract_code_law_entities(query))

    seen: set[str] = set()
    deduped: list[str] = []
    for entity in all_entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(entity)
    return deduped


def merge_legal_entities(
    base_entities: list[str],
    parsed_entities: list[str],
) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []

    for entity in base_entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            merged.append(entity)

    for entity in parsed_entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            merged.append(entity)

    return merged
