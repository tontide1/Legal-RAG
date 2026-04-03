# Checklist triển khai giao diện Streamlit cho Legal-RAG

Mục tiêu của checklist này là triển khai bản **MVP Streamlit 1 trang** theo hướng **minh bạch pipeline** (`NER -> retrieval -> graph rerank -> Gemini`).

## 1) `requirements.txt`

- [ ] Thêm dependency `streamlit`.
- [ ] Giữ nguyên các dependency hiện có, không xóa/sửa version nếu chưa cần.
- [ ] Kiểm tra cài đặt thành công bằng `python3 -m pip install -r requirements.txt`.

## 2) `streamlit_app.py` (file mới, tại repo root)

- [ ] Tạo entrypoint Streamlit riêng, không thay thế `src/main.py`.
- [ ] Thêm bootstrap import path để gọi được `src/legal_qa.py`.
- [ ] Import và gọi `run_legal_qa(query)` làm API chính cho UI.
- [ ] Tạo header mô tả ngắn pipeline và phạm vi MVP.
- [ ] Tạo ô nhập câu hỏi tiếng Việt bằng `st.text_area`.
- [ ] Tạo nút `Tra cứu` để trigger pipeline.
- [ ] Dùng `st.session_state` lưu `query` và `result` gần nhất.
- [ ] Hiển thị block `Câu trả lời` từ `result["answer_text"]`.
- [ ] Hiển thị block `Thực thể NER` từ `result["ner_entities"]`.
- [ ] Hiển thị block `Căn cứ truy xuất` từ `result["retrieved_nodes"]`.
- [ ] Với `retrieved_nodes`, render bảng gồm cột:
  - `label`
  - `name`
  - `value`
  - `bm25`
  - `cosine`
  - `graph_sum`
  - `final_score`
- [ ] Hiển thị block `Citation` từ `result["citations"]`.
- [ ] Hiển thị block `Chẩn đoán` gồm `scores`, `timings`, `errors`.
- [ ] Xử lý trạng thái không có dữ liệu retrieve (abstain) rõ ràng cho người dùng.
- [ ] Xử lý query rỗng bằng cảnh báo thân thiện.
- [ ] Bọc gọi pipeline bằng `try/except` để không crash toàn trang.

## 3) `src/ui_runtime.py` (file mới, adapter tối ưu runtime)

- [ ] Tạo module adapter cho UI để gom logic runtime nặng.
- [ ] Tạo hàm health check env (Neo4j/Gemini) trả về trạng thái có thể hiển thị trên sidebar.
- [ ] Tạo wrapper gọi pipeline theo chuẩn output dict thống nhất cho UI.
- [ ] Nếu cần tối ưu hiệu năng, dùng `st.cache_resource` cho thành phần khởi tạo nặng.
- [ ] Không sao chép lại nghiệp vụ NER/retrieval/generation ở module này.
- [ ] Không phá vỡ invariants hiện tại của `run_legal_qa`.

## 4) `src/legal_qa.py`

- [ ] Chỉ chỉnh khi thực sự cần cho UI integration.
- [ ] Không đổi contract output đang được test bao phủ (`query`, `ner_entities`, `retrieved_nodes`, `context_text`, `answer_text`, `citations`, `scores`, `timings`, `errors`).
- [ ] Nếu thêm tham số tùy chọn cho UI (ví dụ cache injection), giữ backward compatibility cho CLI/tests.
- [ ] Không thay đổi logic cốt lõi pipeline ngoài phạm vi UI integration.

## 5) `src/main.py`

- [ ] Giữ nguyên CLI entrypoint đang có.
- [ ] Chỉ sửa nếu cần đồng bộ nhỏ (ví dụ import path), tránh thay đổi hành vi.

## 6) `README.md`

- [ ] Thêm mục chạy Streamlit local.
- [ ] Bổ sung lệnh:
  - `streamlit run streamlit_app.py`
- [ ] Cập nhật phần biến môi trường bắt buộc cho demo UI.
- [ ] Thêm ghi chú trạng thái lỗi thường gặp:
  - thiếu `GOOGLE_API_KEY`
  - thiếu `NEO4J_PASSWORD`
  - `429 RESOURCE_EXHAUSTED`

## 7) `tests/test_legal_qa.py`

- [ ] Chạy lại test để đảm bảo UI không làm vỡ contract pipeline.
- [ ] Nếu có chỉnh `src/legal_qa.py`, bổ sung test tương ứng.

## 8) `tests/test_streamlit_smoke.py` (tùy chọn, file mới)

- [ ] Thêm smoke test import `streamlit_app.py` không lỗi syntax/import.
- [ ] Mock call `run_legal_qa` để test luồng render cơ bản (nếu triển khai được gọn).
- [ ] Giữ test nhẹ, tránh phụ thuộc Neo4j/Gemini thật.

## 9) Checklist chạy lệnh xác minh

- [ ] Cài dependency:

```bash
python3 -m pip install -r requirements.txt
```

- [ ] Chạy unit test tối thiểu:

```bash
python3 -m unittest tests.test_legal_qa
python3 -m unittest tests.test_pipeline_utils
```

- [ ] Chạy syntax sanity:

```bash
python3 -m py_compile src/main.py src/legal_qa.py streamlit_app.py
```

- [ ] Chạy demo UI local:

```bash
streamlit run streamlit_app.py
```

## 10) Acceptance checklist (MVP)

- [ ] Người dùng nhập câu hỏi tiếng Việt trên web và nhận được `answer_text`.
- [ ] UI hiển thị được `ner_entities`.
- [ ] UI hiển thị được top `retrieved_nodes` với score.
- [ ] UI hiển thị được `citations`.
- [ ] UI hiển thị được `timings` và `errors`.
- [ ] Query rỗng không làm crash app.
- [ ] Trường hợp không có retrieval hiển thị fallback rõ ràng.
- [ ] Không làm hỏng CLI hiện tại (`python3 src/main.py`).

## 11) Phạm vi ngoài checklist (không làm trong vòng MVP này)

- [ ] Không triển khai multi-page phức tạp.
- [ ] Không refactor lớn thuật toán retrieval.
- [ ] Không thay đổi mô hình NER/embedding trong đợt dựng UI.
- [ ] Không mở rộng frontend production-level.
