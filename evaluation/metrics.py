from __future__ import annotations

import math
import re


def _top_k(items: list[str], k: int) -> list[str]:
    return list(items[:k])


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def hit_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return 1.0 if any(node_id in gold_ids for node_id in _top_k(retrieved_ids, k)) else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for node_id in _top_k(retrieved_ids, k) if node_id in gold_ids)
    return hits / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    top_items = _top_k(retrieved_ids, k)
    if not top_items:
        return 0.0
    hits = sum(1 for node_id in top_items if node_id in gold_ids)
    return hits / len(top_items)


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    top_items = _top_k(retrieved_ids, k)
    if not top_items or not gold_ids:
        return 0.0

    dcg = 0.0
    for index, node_id in enumerate(top_items, start=1):
        if node_id in gold_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(gold_ids), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def citation_scores(found_citations: list[str], gold_citations: list[str]) -> dict[str, float]:
    found = {_normalize_text(value) for value in found_citations if value}
    gold = {_normalize_text(value) for value in gold_citations if value}

    if not found:
        precision = 0.0
    else:
        precision = len(found & gold) / len(found)

    if not gold:
        recall = 0.0
    else:
        recall = len(found & gold) / len(gold)

    exact_match = 1.0 if found == gold and gold else 0.0
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_exact_match": exact_match,
    }


def abstention_precision(records: list[dict]) -> float:
    abstained_records = [record for record in records if record.get("did_abstain")]
    if not abstained_records:
        return 0.0
    correct = sum(1 for record in abstained_records if record.get("answerable") is False)
    return correct / len(abstained_records)
