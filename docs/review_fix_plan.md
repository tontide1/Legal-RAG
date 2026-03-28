# Kế hoạch sửa lỗi Legal-RAG sau code review

## Tóm tắt
Tài liệu này tổng hợp các lỗi đã được review, mức độ ưu tiên và hướng sửa đã triển khai để ổn định pipeline Legal-RAG hiện tại.

## Ưu tiên cao
| Vấn đề | Tác động | Hướng sửa |
| --- | --- | --- |
| `main.py` trỏ sai đường dẫn model NER | Entry point không chạy được | Chuẩn hóa `CODE_ROOT` và trỏ model về `Code/NER/bilstm_ner.pt` |
| Graph embedding dùng `ten` làm khóa | Node trùng tên giữa nhiều văn bản bị đè embedding | Thêm `node_id = "{Label}::{Tên}"` và dùng xuyên suốt Neo4j, embedding, retrieval |
| `save_data.py` xóa toàn bộ Neo4j | Rủi ro mất dữ liệu ngoài app | Chỉ xóa node mang label `LegalRAG` |

## Ưu tiên trung bình
| Vấn đề | Tác động | Hướng sửa |
| --- | --- | --- |
| `Value` rỗng bị biến thành `None` trong text retrieval | Nhiễu BM25 và embedding | Chuẩn hóa text payload để `None` thành chuỗi rỗng |
| Graph rerank cắt `top_k` quá sớm | Graph signal không cứu được candidate ngoài cutoff | Thêm `candidate_pool_size`, rerank từ pool lớn hơn |
| Module có side effect khi import | Khó test, dễ chạy nhầm tác vụ nặng | Chuyển sang `main()` và guard `if __name__ == "__main__"` |

## Thứ tự thực hiện
1. Ổn định runtime ở `main.py`
2. Chuẩn hóa định danh node trong Neo4j
3. Làm import dữ liệu an toàn hơn
4. Nâng chất lượng retrieval/rerank
5. Tách side effect khỏi import

## Checklist xác minh
- `python Code/main.py` đi qua được bước resolve model NER
- Neo4j lưu node theo `node_id`, không còn đè node trùng `ten`
- Chỉ dữ liệu `LegalRAG` bị xóa khi import lại
- Retrieval rerank từ pool lớn hơn `top_k`
- Import module không tự động train hoặc gọi API

## Ghi chú môi trường
Lần sửa này vẫn phụ thuộc vào việc environment `RAG` có đủ package như `python-dotenv`, `torch`, `neo4j`, `sentence-transformers`, `langchain-google-genai`.
