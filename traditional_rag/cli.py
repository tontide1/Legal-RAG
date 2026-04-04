from __future__ import annotations

import argparse
import json
from pathlib import Path

from traditional_rag.pipeline import TraditionalRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy traditional RAG cơ bản trên corpus văn bản pháp luật."
    )
    parser.add_argument("--query", required=True, help="Câu hỏi người dùng cần trả lời.")
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Thư mục nguồn dữ liệu văn bản để xây dựng chỉ mục.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Số tài liệu top được trả về làm ngữ cảnh.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Tùy chọn: lưu kết quả ra file JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag = TraditionalRAG(dataset_root=args.dataset_root, top_k=args.top_k)
    result = rag.run(args.query)

    print("\n=== Traditional RAG Result ===")
    print(f"Query: {result['query']}")
    print(f"Top documents: {result['scores']['retrieved_count']}")
    print("\nAnswer:\n")
    print(result["answer"])
    print("\n=== Retrieved Sources ===")
    for idx, doc in enumerate(result["retrieved_documents"], start=1):
        print(
            f"{idx}. {doc['title']} ({doc['doc_id']}) - combined_score={doc['combined_score']:.4f}"
        )

    if args.output_json:
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved output to {args.output_json}")


if __name__ == "__main__":
    main()
