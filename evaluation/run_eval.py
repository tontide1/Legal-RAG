from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.judge import run_judge_metrics
from evaluation.metrics import (
    abstention_precision,
    citation_scores,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evaluation.report import build_summary_markdown, write_json, write_jsonl, write_markdown
from src.legal_qa import DEFAULT_ABSTAIN_ANSWER, run_legal_qa


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_records(records: list[dict], *, enable_llm_judge: bool) -> tuple[list[dict], dict]:
    per_query = []
    abstention_rows = []
    hit_scores = []
    recall_scores = []
    precision_scores = []
    ndcg_scores = []
    citation_precision_scores = []
    citation_recall_scores = []
    citation_exact_scores = []

    for record in records:
        result = run_legal_qa(record["query"])
        retrieved_ids = [node.get("node_id") for node in result["retrieved_nodes"] if node.get("node_id")]
        gold_ids = set(record.get("gold_node_ids", []))
        found_citations = [citation.get("name", "") for citation in result["citations"]]
        gold_citations = record.get("gold_citations", [])
        citation_metric_row = citation_scores(found_citations, gold_citations)

        hit_scores.append(hit_at_k(retrieved_ids, gold_ids, 5))
        recall_scores.append(recall_at_k(retrieved_ids, gold_ids, 5))
        precision_scores.append(precision_at_k(retrieved_ids, gold_ids, 5))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, gold_ids, 10))
        citation_precision_scores.append(citation_metric_row["citation_precision"])
        citation_recall_scores.append(citation_metric_row["citation_recall"])
        citation_exact_scores.append(citation_metric_row["citation_exact_match"])

        did_abstain = result["answer_text"] in {
            DEFAULT_ABSTAIN_ANSWER,
        }
        abstention_rows.append(
            {
                "answerable": record.get("answerable", True),
                "did_abstain": did_abstain,
            }
        )

        per_query.append(
            {
                "id": record.get("id"),
                "query": record["query"],
                "category": record.get("category"),
                "answerable": record.get("answerable", True),
                "result": result,
                "metrics": {
                    "hit_at_5": hit_scores[-1],
                    "recall_at_5": recall_scores[-1],
                    "precision_at_5": precision_scores[-1],
                    "ndcg_at_10": ndcg_scores[-1],
                    **citation_metric_row,
                },
            }
        )

    query_count = len(records)
    summary = {
        "query_count": query_count,
        "metrics": {
            "hit_at_5": sum(hit_scores) / query_count if query_count else 0.0,
            "recall_at_5": sum(recall_scores) / query_count if query_count else 0.0,
            "precision_at_5": sum(precision_scores) / query_count if query_count else 0.0,
            "ndcg_at_10": sum(ndcg_scores) / query_count if query_count else 0.0,
            "citation_precision": sum(citation_precision_scores) / query_count if query_count else 0.0,
            "citation_recall": sum(citation_recall_scores) / query_count if query_count else 0.0,
            "citation_exact_match": sum(citation_exact_scores) / query_count if query_count else 0.0,
            "abstention_precision": abstention_precision(abstention_rows),
        },
    }
    summary.update(run_judge_metrics(enabled=enable_llm_judge))
    return per_query, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic evaluation for Legal-RAG.")
    parser.add_argument(
        "--input",
        default="evaluation/fixtures/mini_eval.jsonl",
        help="Path to the benchmark JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/reports/latest",
        help="Directory to store summary and per-query reports.",
    )
    parser.add_argument(
        "--disable-llm-judge",
        action="store_true",
        help="Skip LLM judge metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    records = load_jsonl(input_path)
    per_query, summary = evaluate_records(records, enable_llm_judge=not args.disable_llm_judge)

    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "summary.md", build_summary_markdown(summary))
    write_jsonl(output_dir / "per_query.jsonl", per_query)


if __name__ == "__main__":
    main()
