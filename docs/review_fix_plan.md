# Tổng hợp sửa lỗi Legal-RAG sau code review

## Tóm tắt
Tài liệu này tổng hợp các lỗi đã được review, các thay đổi đã triển khai để ổn định codebase, và các điểm còn mở cần lưu ý khi vận hành.

## Đã triển khai
| Vấn đề | Tác động | Hướng sửa |
| --- | --- | --- |
| `main.py` trỏ sai đường dẫn model NER | Entry point không chạy được | Chuẩn hóa `CODE_ROOT` và trỏ model về `Code/NER/bilstm_ner.pt` |
| Graph embedding dùng `ten` làm khóa | Node trùng tên giữa nhiều văn bản bị đè embedding | Thêm `node_id = "{Label}::{Tên}"` và dùng xuyên suốt Neo4j, embedding, retrieval |
| `save_data.py` xóa toàn bộ Neo4j | Rủi ro mất dữ liệu ngoài app | Chỉ xóa node mang label `LegalRAG` |
| `Value` rỗng bị biến thành `None` trong text retrieval | Nhiễu BM25 và embedding | Chuẩn hóa text payload để `None` thành chuỗi rỗng |
| Graph rerank cắt `top_k` quá sớm | Graph signal không cứu được candidate ngoài cutoff | Thêm `candidate_pool_size`, rerank từ pool lớn hơn |
| Module có side effect khi import | Khó test, dễ chạy nhầm tác vụ nặng | Chuyển sang `main()` và guard `if __name__ == "__main__"` |
| Gemini model hardcode trong source | Khó đổi model khi vận hành | Đọc từ `GEMINI_MODEL`, mặc định `gemini-2.5-flash-lite` |

## Checklist xác minh
- `python Code/main.py` đi qua được bước resolve model NER
- Neo4j lưu node theo `node_id`, không còn đè node trùng `ten`
- Chỉ dữ liệu `LegalRAG` bị xóa khi import lại
- Retrieval rerank từ pool lớn hơn `top_k`
- Import module không tự động train hoặc gọi API
- Có thể đổi model Gemini qua `GEMINI_MODEL` mà không sửa source

## Điểm còn mở
- NER hiện chủ yếu nhận tham chiếu `Điều <số>`; chưa nhận tốt tên luật/văn bản như `Luật Hải quan`
- Nếu Gemini API key hết quota sẽ vẫn có thể gặp `429 RESOURCE_EXHAUSTED`
- Chưa có test end-to-end với Neo4j + Gemini thật

## Ghi chú môi trường
Lần sửa này vẫn phụ thuộc vào việc environment `RAG` có đủ package như `python-dotenv`, `torch`, `neo4j`, `sentence-transformers`, `langchain-google-genai`.

## Cấu hình Gemini
- Thêm biến môi trường `GEMINI_MODEL`
- Giá trị mặc định hiện tại: `gemini-2.5-flash-lite`
- Có thể override trong `.env` nếu cần đổi model mà không sửa source
