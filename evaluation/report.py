from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def build_summary_markdown(summary: dict) -> str:
    metrics = summary.get("metrics", {})
    lines = [
        "# Legal-RAG Evaluation Summary",
        "",
        f"- Queries: {summary.get('query_count', 0)}",
        f"- Judge skipped: {summary.get('judge_metrics_skipped', True)}",
        f"- Hit@5: {metrics.get('hit_at_5', 0.0):.3f}",
        f"- Recall@5: {metrics.get('recall_at_5', 0.0):.3f}",
        f"- Precision@5: {metrics.get('precision_at_5', 0.0):.3f}",
        f"- nDCG@10: {metrics.get('ndcg_at_10', 0.0):.3f}",
        f"- Citation precision: {metrics.get('citation_precision', 0.0):.3f}",
        f"- Abstention precision: {metrics.get('abstention_precision', 0.0):.3f}",
    ]
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
