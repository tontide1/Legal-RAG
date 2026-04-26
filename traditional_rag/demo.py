#!/usr/bin/env python3
"""
Demo script for Traditional RAG pipeline.
This script demonstrates the basic functionality without requiring all dependencies.
"""

from pathlib import Path

def demo_chunking():
    """Demo basic text chunking functionality."""
    print("=== Text Chunking Demo ===")

    # Simple text chunking without NLTK
    def simple_chunk_text(text: str, chunk_size: int = 500) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        return chunks

    sample_text = """
    Điều 1: Phạm vi điều chỉnh. Luật này điều chỉnh quan hệ xã hội phát sinh từ hoạt động thương mại;
    quyền, nghĩa vụ của thương nhân, tổ chức, cá nhân trong hoạt động thương mại.
    Luật này cũng áp dụng cho hoạt động đầu tư kinh doanh, thương mại có yếu tố nước ngoài trên lãnh thổ Việt Nam.

    Điều 2: Đối tượng áp dụng. Luật này áp dụng đối với thương nhân, tổ chức, cá nhân Việt Nam và nước ngoài
    tham gia hoạt động thương mại, đầu tư kinh doanh trên lãnh thổ Việt Nam.
    """

    chunks = simple_chunk_text(sample_text, chunk_size=300)
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} (length: {len(chunk)}):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

def demo_file_loading():
    """Demo loading documents from dataset."""
    print("\n=== Document Loading Demo ===")

    dataset_root = Path("dataset")
    if not dataset_root.exists():
        print(f"Dataset directory '{dataset_root}' does not exist.")
        print("Creating sample documents for demo...")

        # Create sample directory and files
        dataset_root.mkdir(exist_ok=True)
        sample_dir = dataset_root / "sample"
        sample_dir.mkdir(exist_ok=True)

        # Create sample legal documents
        sample_docs = {
            "luat_thuong_mai.txt": """
Điều 1: Phạm vi điều chỉnh
Luật này điều chỉnh các quan hệ xã hội phát sinh từ hoạt động thương mại; quyền, nghĩa vụ của thương nhân,
tổ chức, cá nhân trong hoạt động thương mại.

Điều 2: Giải thích từ ngữ
Trong Luật này, các từ ngữ dưới đây được hiểu như sau:
1. Thương nhân là cá nhân, pháp nhân có đăng ký kinh doanh theo quy định của pháp luật.
2. Hoạt động thương mại là việc mua bán hàng hóa, cung ứng dịch vụ.
""",
            "nghi_dinh_100.txt": """
Điều 1: Phạm vi áp dụng
Nghị định này quy định chi tiết thi hành Luật Thương mại về đăng ký kinh doanh.

Điều 2: Hồ sơ đăng ký
Hồ sơ đăng ký kinh doanh bao gồm:
1. Giấy đề nghị đăng ký kinh doanh.
2. Bản sao giấy tờ chứng thực cá nhân của chủ sở hữu.
3. Bản sao giấy tờ chứng thực địa chỉ kinh doanh.
"""
        }

        for filename, content in sample_docs.items():
            (sample_dir / filename).write_text(content.strip(), encoding='utf-8')

    # Load documents
    documents = []
    for path in sorted(dataset_root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                documents.append({
                    "doc_id": str(path.relative_to(dataset_root)),
                    "title": path.stem,
                    "source": str(path),
                    "text": text,
                })
        except OSError:
            continue

    print(f"Loaded {len(documents)} documents:")
    for doc in documents:
        print(f"- {doc['title']}: {len(doc['text'])} characters")

    return documents

def main():
    """Main demo function."""
    print("Traditional RAG Pipeline Demo")
    print("=" * 40)

    # Demo chunking
    demo_chunking()

    # Demo file loading
    documents = demo_file_loading()

    print("\n=== Summary ===")
    print(f"Total documents loaded: {len(documents)}")
    print("Pipeline components:")
    print("- TextChunker: Splits documents into manageable chunks")
    print("- TextEmbedder: Creates vector embeddings for semantic search")
    print("- LLMResponseGenerator: Generates answers using retrieved context")
    print("- TraditionalRAGPipeline: Orchestrates the entire RAG process")

    print("\nTo run the full pipeline, install dependencies from requirements.txt")
    print("and set GOOGLE_API_KEY environment variable.")

if __name__ == "__main__":
    main()