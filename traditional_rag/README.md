# Traditional RAG

Thư mục này chứa một triển khai RAG truyền thống cơ bản cho hệ thống pháp luật.

## Mục tiêu

- Thu thập tài liệu văn bản từ `dataset/`.
- Xây dựng chỉ mục BM25 + embedding truy vấn bằng `SentenceTransformer`.
- Truy xuất top-k tài liệu liên quan.
- Kết hợp ngữ cảnh để tạo câu trả lời với Gemini.

## Cách sử dụng

1. Cài đặt các phụ thuộc trong `requirements.txt`.
2. Đặt biến môi trường `GOOGLE_API_KEY` trong `.env`.
3. Chạy:

```bash
python traditional_rag/cli.py --query "Hỏi về một điều luật cụ thể" --top-k 3
```

## File chính

- `traditional_rag/pipeline.py`: pipeline RAG truyền thống.
- `traditional_rag/cli.py`: entrypoint CLI đơn giản.
- `traditional_rag/__init__.py`: export module.
