# Traditional RAG Pipeline for Vietnamese Legal Documents

This directory contains a complete, standalone traditional RAG (Retrieval-Augmented Generation) pipeline. It is designed to answer questions about Vietnamese legal documents using a classic approach: Load -> Chunk -> Embed -> Store -> Retrieve -> Generate.

This implementation does **not** use the graph-based components from the main project.

## Luồng hoạt động

Pipeline hoạt động theo các bước tuần tự sau:

1.  **Chuẩn bị dữ liệu (Một lần duy nhất)**:
    *   **Load & Chunk**: Tệp `chunking.py` đọc tất cả các tệp văn bản (`.txt`) từ thư mục `dataset` ở gốc dự án. Các tài liệu lớn được chia thành các đoạn nhỏ hơn (chunks).
    *   **Embed & Store**: Tệp `embedding.py` lấy các chunks này, sử dụng mô hình `keepitreal/vietnamese-sbert` để chuyển đổi mỗi chunk thành một vector embedding. Tất cả các vector này sau đó được lưu vào một cơ sở dữ liệu vector (Vector Store) bằng FAISS tại `traditional_rag/vector_store/`.

2.  **Hỏi-đáp (Mỗi khi có câu hỏi)**:
    *   **Retrieve**: Khi bạn chạy `pipeline.py` với một câu hỏi, nó sẽ tải Vector Store đã được tạo. Câu hỏi của bạn cũng được chuyển thành vector và được sử dụng để tìm kiếm 5 chunks có nội dung liên quan nhất trong Vector Store.
    *   **Generate**: Câu hỏi gốc cùng với 5 chunks văn bản liên quan được gửi đến một Mô hình Ngôn ngữ Lớn (LLM) của Google (Gemini).
    *   **Response**: LLM tổng hợp thông tin và tạo ra câu trả lời cuối cùng cho bạn.

## Cách sử dụng

### 1. Cài đặt thư viện

Mở terminal và chạy lệnh sau từ thư mục gốc của dự án để cài đặt tất cả các thư viện cần thiết:

```bash
pip install -r traditional_rag/requirements.txt
```

### 2. Đặt câu hỏi

Sau khi bước hoàn tất, có thể bắt đầu đặt câu hỏi. Sử dụng lệnh sau:

```bash
python -m traditional_rag.pipeline
```

Sau đó nhập câu hỏi:

```bash
"Nhiệm vụ của hải quan Việt Nam được quy định tại Điều 12 Luật Hải quan là gì?"
```

Hệ thống sẽ xử lý và in ra câu trả lời trong terminal.
