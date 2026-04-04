import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from legal_qa import run_legal_qa


def main():
    query = input("Nhập query của bạn: ").strip()
    result = run_legal_qa(query)
    if result["errors"] and not result["query"]:
        print(result["errors"][0])
        return

    print("\nThực thể được NER nhận diện:", result["ner_entities"])

    if not result["ner_entities"]:
        print("\nKhông nhận diện được thực thể nào. Truy vấn toàn câu:")

    if not result["retrieved_nodes"]:
        print("Không tìm thấy căn cứ pháp lý phù hợp.")
        print(result["answer_text"])
        return

    print("\n=== Căn cứ truy xuất ===")
    for index, node in enumerate(result["retrieved_nodes"], start=1):
        score = node.get("final_score", node.get("combined_score", 0.0))
        print(f"{index}. {node['name']} - {node.get('value', '')} (Label: {node.get('label')}) - (Score: {score:.4f})")

    print("\n=== Câu trả lời từ LLM ===")
    print(result["answer_text"])

    if result["citations"]:
        print("\n=== Citation ===")
        for citation in result["citations"]:
            print(f"- {citation['label']} ({citation['name']})")

    if result["errors"]:
        print("\n=== Cảnh báo ===")
        for error in result["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
