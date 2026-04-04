# Simple Legal RAG - Hỏi đáp Pháp luật Việt Nam

## Tổng quan

Hệ thống RAG (Retrieval-Augmented Generation) đơn giản cho văn bản pháp luật Việt Nam.

**Pipeline:**

```
Câu hỏi -> [BM25 + Semantic Search] -> RRF Merge -> Top-K Chunks -> Gemini -> Câu trả lời
```

### Kiến trúc

```
+-------------+      +----------------+
|  Câu hỏi   |----->| BM25 Lexical   |--+
|  tiếng Việt |      +----------------+  |   +--------------+   +----------+   +--------------+
|             |                          +-->| RRF Merge    |-->| Top-K    |-->| Gemini API   |
|             |      +----------------+  |   +--------------+   | Chunks   |   | -> Trả lời   |
|             |----->| Vietnamese     |--+                      +----------+   +--------------+
|             |      | SBERT Semantic |
+-------------+      +----------------+
```

## Cấu trúc thư mục

```
SimpleRAG/
|-- dataset/           
|   |-- Luật/
|   |-- Nghị_Định/
|   |-- Nghị_Quyết/
|   |-- Pháp_Lệnh/
|   +-- Thông_tư/
|-- chroma_db/            # Vector store (tự tạo sau khi ingest)
|-- ingest.py             # Script 1: Đọc + chunk + embedding + lưu ChromaDB
|-- query.py              # Script 2: Interactive QA (BM25 + semantic + Gemini)
|-- requirements.txt      # Dependencies
+-- RAGSIMPLE.md         
```

## Các thành phần chính

### 1. Chunking theo cấu trúc Điều (ingest.py)

- Tự động nhận diện ranh giới "Điều <số>" trong văn bản pháp luật
- Mỗi Điều = 1 chunk (giữ nguyên ngữ cảnh pháp lý)
- Điều quá dài (> 2000 ký tự) -> tự split với overlap 200 ký tự
- Phần mở đầu văn bản (trước Điều 1) -> chunk riêng "Mở đầu"

### 2. Embedding

- Model: keepitreal/vietnamese-sbert (768 dimensions)
- Được thiết kế cho văn bản tiếng Việt
- Cùng model cho cả document embedding và query embedding

### 3. Vector Store - ChromaDB

- Lưu local tại chroma_db/
- Collection: legal_chunks
- Distance metric: cosine
- Metadata: doc_name, dieu, char_len

### 4. Hybrid Retrieval (query.py)

| Phương pháp  | Mô tả                                                              | Top-K  |
|--------------|---------------------------------------------------------------------|--------|
| BM25         | Lexical search - tốt cho truy vấn chứa tên Điều, số hiệu cụ thể   | 15     |
| Semantic     | Vietnamese SBERT qua ChromaDB - tốt cho câu hỏi diễn đạt lại ý    | 15     |
| RRF Merge    | Reciprocal Rank Fusion kết hợp 2 ranked list                       | Top 7  |

### 5. Answer Generation

- Model: Gemini (GEMINI_MODEL từ .env, mặc định gemini-2.5-flash-lite)
- System prompt yêu cầu:
  - Chỉ trả lời dựa trên evidence
  - Trích dẫn cụ thể Điều/Khoản/Điểm
  - Trả lời tiếng Việt
- Temperature: 0.2 (bảo thủ, phù hợp pháp luật)

---

## Hướng dẫn chạy

### Yêu cầu

- Python 3.11+
- Conda env RAG (hoặc bất kỳ env nào)
- KHÔNG CẦN tạo .env riêng. SimpleRAG tự động đọc file Legal-RAG/.env (thư mục cha)
  File .env cần có:
  ```
  GOOGLE_API_KEY="..."
  GEMINI_MODEL=gemini-2.5-flash-lite
  ```

### Bước 1: Cài dependencies

```bash
conda activate RAG
cd SimpleRAG
pip install -r requirements.txt
```

### Bước 2: Ingest dữ liệu (chỉ cần chạy 1 lần, hoặc khi dataset thay đổi)

```bash
python ingest.py
```

Output mẫu:
```
[INFO] Tìm thấy 5 file .txt
   - Luật Hải quan/54.2014.QH13.txt
   - Luật Xử lý vi phạm hành chính/VanBanGoc_15.2012.QH13.txt
   ...
   [CHUNK] Luật Hải quan/54.2014.QH13.txt: 105 chunk(s)
   ...
[TONG] 450 chunks
[MODEL] Đang tải model embedding: keepitreal/vietnamese-sbert ...
[EMBED] Đang tạo embedding ...
[OK] Đã tạo 450 embedding (dim=768)
[SAVE] Lưu vào ChromaDB ...
[DONE] Hoàn tất!
```

### Bước 3: Hỏi đáp

```bash
python query.py
```

Ví dụ tương tác:
```
============================================================
  SIMPLE LEGAL RAG - Hỏi đáp pháp luật Việt Nam
============================================================
[OK] Retriever sẵn sàng!

Nhập câu hỏi (gõ 'exit' hoặc 'q' để thoát):

Câu hỏi: Thời hiệu xử phạt vi phạm hành chính là bao lâu?
[SEARCH] Đang tìm kiếm ...
[RESULT] Tìm thấy 7 trích đoạn liên quan:
   1. [Điều 6] Luật sửa đổi VPHC/67_2020_QH14_m_373520.txt: ...

[LLM] Đang tạo câu trả lời ...

============================================================
TRẢ LỜI:
------------------------------------------------------------
Theo Điều 6, thời hiệu xử phạt VPHC là 01 năm, trừ một số
lĩnh vực đặc biệt là 02 năm...
============================================================
```

---

## So sánh với Graph RAG (dự án chính)

| Tiêu chí      | Simple RAG              | Graph RAG (src/)                  |
|---------------|-------------------------|-----------------------------------|
| Vector Store  | ChromaDB (local)        | Neo4j                             |
| Chunking      | Theo Điều (regex)       | Entity-based                      |
| Retrieval     | BM25 + SBERT -> RRF     | BM25 + SBERT + Graph Rerank       |
| NER           | Không                   | BiLSTM (B-ARTICLE, I-ARTICLE)     |
| Graph         | Không                   | Neo4j Knowledge Graph             |
| Reranking     | RRF đơn giản            | Graph-aware reranking             |
| LLM           | Gemini                  | Gemini                            |
| Độ phức tạp   | Thấp                    | Cao                               |
| Phù hợp      | Demo, prototype         | Production, nghiên cứu            |

---

## Lưu ý kỹ thuật

1. Embedding space: Query và document phải dùng cùng model (keepitreal/vietnamese-sbert). Nếu đổi model, phải chạy lại ingest.py.
2. ChromaDB lưu tại chroma_db/ - xóa thư mục này nếu muốn reset hoàn toàn.
3. Thứ tự chạy: Luôn chạy ingest.py trước query.py.
4. Dataset: Chỉ đọc file .txt. Nếu thêm/sửa file .txt trong dataset/, chạy lại ingest.py.
5. File .env: KHÔNG cần tạo riêng. query.py tự động đọc từ Legal-RAG/.env (thư mục cha).
