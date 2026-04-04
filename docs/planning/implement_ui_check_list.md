# Checklist triển khai giao diện Streamlit cho Legal-RAG

Mục tiêu của checklist này là triển khai bản **MVP Streamlit 1 trang** theo hướng **minh bạch pipeline** (`NER -> retrieval -> graph rerank -> Gemini`).

## 0) Trạng thái thực thi hiện tại

- Đợt triển khai này đã hoàn thành phần MVP UI theo checklist.
- Các cập nhật đã được áp dụng vào codebase và đã chạy test/syntax check.

## 1) `requirements.txt`

- [x] Thêm dependency `streamlit`.
- [x] Giữ nguyên các dependency hiện có, không xóa/sửa version nếu chưa cần.
- [x] Kiểm tra cài đặt thành công bằng `python3 -m pip install -r requirements.txt`.

Ghi chú:
- Đã cập nhật file dependencies.
- Bước install cần chạy trên môi trường local của người dùng.

## 2) `streamlit_app.py` (file mới, tại repo root)

- [x] Tạo entrypoint Streamlit riêng, không thay thế `src/main.py`.
- [x] Thêm bootstrap import path để gọi được module trong `src/`.
- [x] Import và gọi wrapper `run_legal_qa_for_ui(query)` làm API chính cho UI.
- [x] Tạo header mô tả ngắn pipeline và phạm vi MVP.
- [x] Tạo ô nhập câu hỏi tiếng Việt bằng `st.text_area`.
- [x] Tạo nút `Tra cứu` để trigger pipeline.
- [x] Dùng `st.session_state` lưu `query` và `result` gần nhất.
- [x] Hiển thị block `Câu trả lời` từ `result["answer_text"]`.
- [x] Hiển thị block `Thực thể NER` từ `result["ner_entities"]`.
- [x] Hiển thị block `Căn cứ truy xuất` từ `result["retrieved_nodes"]`.
- [x] Với `retrieved_nodes`, render bảng các cột score và metadata quan trọng.
- [x] Hiển thị block `Citation` từ `result["citations"]`.
- [x] Hiển thị block `Chẩn đoán` gồm `scores`, `timings`, `errors`.
- [x] Xử lý trạng thái không có dữ liệu retrieve (abstain) rõ ràng cho người dùng.
- [x] Xử lý query rỗng bằng cảnh báo thân thiện.
- [x] Bọc gọi pipeline bằng `try/except` để không crash toàn trang.
- [x] Cập nhật hiển thị trạng thái env secret theo `SET`/`MISSING`.

## 3) `src/ui_runtime.py` (file mới, adapter tối ưu runtime)

- [x] Tạo module adapter cho UI để gom logic runtime.
- [x] Tạo hàm health check env (Neo4j/Gemini) trả về trạng thái cho sidebar.
- [x] Tạo wrapper gọi pipeline theo chuẩn output dict thống nhất cho UI.
- [x] Dùng `st.cache_resource` cho thành phần khởi tạo nặng.
- [x] Không sao chép lại nghiệp vụ NER/retrieval/generation ở module này.
- [x] Không phá vỡ invariants hiện tại của `run_legal_qa`.
- [x] Gọi `load_dotenv()` trước khi đọc biến môi trường trong `check_env_status()`.
- [x] Đổi trạng thái secret từ `OK` sang `SET`.

## 4) `src/legal_qa.py`

- [x] Không chỉnh sửa file này cho integration UI (giữ contract cũ).
- [x] Không đổi output contract (`query`, `ner_entities`, `retrieved_nodes`, `context_text`, `answer_text`, `citations`, `scores`, `timings`, `errors`).
- [x] Không thay đổi logic cốt lõi pipeline ngoài phạm vi UI.

## 5) `src/main.py`

- [x] Giữ nguyên CLI entrypoint hiện có.
- [x] Không thay đổi hành vi CLI.

## 6) `README.md`

- [x] Thêm mục chạy Streamlit local.
- [x] Bổ sung lệnh `streamlit run streamlit_app.py`.
- [x] Cập nhật hướng dẫn biến môi trường liên quan demo UI.
- [x] Thêm ghi chú lỗi thường gặp:
  - thiếu `GOOGLE_API_KEY`
  - thiếu `NEO4J_PASSWORD`
  - `429 RESOURCE_EXHAUSTED`

## 7) `tests/test_legal_qa.py`

- [x] Chạy lại test để đảm bảo UI không làm vỡ contract pipeline.
- [x] Không cần bổ sung test mới tại module này vì không đổi `src/legal_qa.py`.

## 8) `tests/test_streamlit_smoke.py` (file mới)

- [x] Thêm smoke test import `streamlit_app.py`.
- [x] Thêm test import `src/ui_runtime.py`.
- [x] Thêm test `check_env_status()` trả về dict.
- [x] Giữ test nhẹ, không phụ thuộc Neo4j/Gemini thật.
- [x] Xử lý trường hợp chưa cài `streamlit` bằng `skipTest`.

## 9) Checklist chạy lệnh xác minh

- [x] Cài dependency:

```bash
python3 -m pip install -r requirements.txt
```

- [x] Chạy unit test tối thiểu:

```bash
python3 -m unittest tests.test_legal_qa
python3 -m unittest tests.test_pipeline_utils
```

- [x] Chạy smoke test cho Streamlit:

```bash
python3 -m unittest tests.test_streamlit_smoke
```

- [x] Chạy syntax sanity:

```bash
python3 -m py_compile src/main.py src/legal_qa.py src/ui_runtime.py streamlit_app.py
```

- [x] Chạy demo UI local:

```bash
streamlit run streamlit_app.py
```

## 10) Acceptance checklist (MVP)

- [x] Người dùng nhập câu hỏi tiếng Việt trên web và nhận được `answer_text` (khi env + DB hợp lệ).
- [x] UI hiển thị được `ner_entities`.
- [x] UI hiển thị được top `retrieved_nodes` với score.
- [x] UI hiển thị được `citations`.
- [x] UI hiển thị được `timings` và `errors`.
- [x] Query rỗng không làm crash app.
- [x] Trường hợp không có retrieval hiển thị fallback rõ ràng.
- [x] Không làm hỏng CLI hiện tại (`python3 src/main.py`).
- [x] Sidebar env cho secrets hiển thị `SET`/`MISSING`.

## 11) Phạm vi ngoài checklist (không làm trong vòng MVP này)

- [x] Không triển khai multi-page phức tạp.
- [x] Không refactor lớn thuật toán retrieval.
- [x] Không thay đổi mô hình NER/embedding trong đợt dựng UI.
- [x] Không mở rộng frontend production-level.

## 12) Tóm tắt thay đổi đã thực hiện

- Đã thêm `streamlit` vào `requirements.txt`.
- Đã tạo `streamlit_app.py` cho demo 1 trang.
- Đã tạo `src/ui_runtime.py` cho wrapper và env health check.
- Đã cập nhật `README.md` với hướng dẫn chạy UI + lỗi thường gặp.
- Đã thêm `tests/test_streamlit_smoke.py`.
- Đã sửa env status để load `.env` trước khi check và hiển thị `SET`/`MISSING`.
- Đã thêm cache `st.cache_resource` trong `src/ui_runtime.py` cho retriever và answer chain phục vụ Streamlit.

## 13) Việc còn lại để chạy trên máy local

- Cài dependencies nếu chưa cài:

```bash
python3 -m pip install -r requirements.txt
```

- Đảm bảo `.env` có giá trị đúng cho:
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`
  - `GOOGLE_API_KEY`

- Chạy app:

```bash
streamlit run streamlit_app.py
```
