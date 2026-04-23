# Session Handoff

## Mục tiêu của file
File này lưu trạng thái làm việc mới nhất để session sau có thể đọc nhanh và biết:
- đã làm xong gì
- còn dang dở gì
- nên làm gì tiếp theo
- cần kiểm tra lại bằng cách nào

## Đã hoàn thành trong repo
- Ổn định entrypoint QA:
  - sửa đường dẫn model NER trong `src/main.py`
  - tách cấu hình Gemini sang `src/pipeline_utils.py`
  - model mặc định hiện tại là `gemini-2.5-flash-lite`
- Chuẩn hóa dữ liệu Neo4j:
  - dùng `node_id` làm khóa ổn định
  - label ứng dụng là `LegalRAG`
  - giới hạn xóa dữ liệu trong `save_data.py` vào phạm vi app, không xóa toàn DB
- Sửa retrieval:
  - chuẩn hóa `Value` rỗng thành chuỗi rỗng
  - retrieval và graph embedding dùng `node_id`
  - graph rerank chạy trên `candidate_pool_size` lớn hơn `top_k`
- Giảm side effects:
  - các script nặng đã được đưa về `main()` và guard `if __name__ == "__main__":`
- Thiết lập tài liệu vận hành:
  - `README.md`
  - `docs/review_fix_plan.md`
  - `docs/evaluation_metrics_plan.md`
  - `docs/mvp_plan.md`
  - `docker-compose.yml` cho Neo4j local
- Đồng bộ tài liệu theo layout hiện tại:
  - cập nhật tài liệu chính từ `Code/` sang `src/`
  - thêm liên kết chéo giữa `README.md`, `docs/review_fix_plan.md`, `docs/evaluation_metrics_plan.md`, `docs/mvp_plan.md`
  - cập nhật `AGENTS.md` để phản ánh đúng cấu trúc repo hiện tại
- Thiết lập test tối thiểu:
  - `tests/test_pipeline_utils.py`
  - smoke check bằng `python -m unittest tests.test_pipeline_utils`

## Trạng thái hiện tại cần nhớ
- Repo đã có thêm tài liệu `docs/mvp_plan.md` để chốt hướng MVP cho demo và báo cáo.
- Bộ tài liệu vận hành chính đã được đồng bộ với cấu trúc `src/`.
- Neo4j local có thể chạy bằng Docker qua `docker compose up -d`.
- `.env` được kỳ vọng có các biến:
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`
  - `GOOGLE_API_KEY`
  - `GEMINI_MODEL`
- Retrieval hiện vẫn phụ thuộc vào `keepitreal/vietnamese-sbert`.
- NER hiện chỉ mạnh cho thực thể kiểu `Điều <số>`; chưa xử lý tốt tên luật như `Luật Hải quan`.
- Gemini có thể gặp `429 RESOURCE_EXHAUSTED` nếu API key hết quota hoặc rate limit.
- Tại thời điểm cập nhật handoff này, repo còn thay đổi tài liệu chưa commit.
- `docs/research_NAGphormer_legal_chatbot.md` đang có thay đổi sẵn từ trước, chưa được chỉnh trong lượt cập nhật tài liệu này.

## Việc còn mở
- Chưa có evaluation framework chạy thật theo plan trong `docs/evaluation_metrics_plan.md`.
- Chưa có benchmark dataset `evaluation/data/legal_rag_eval_v1.jsonl`.
- Chưa có đo lường retrieval/generation/citation bằng số liệu thực tế.
- Chưa có test end-to-end với Neo4j + Gemini thật.
- Chưa có cải tiến cho NER hoặc parser để nhận tên luật/văn bản ngoài `Điều <số>`.
- `tests/test_pipeline_utils.py` vẫn cần sửa import để đồng bộ hoàn toàn với layout `src/`.
- Chưa refactor pipeline thành entrypoint callable dùng chung cho CLI, web demo và evaluator.

## Việc nên làm tiếp theo
1. Bám theo `docs/mvp_plan.md` để triển khai MVP ưu tiên demo web và độ đúng đầu ra.
2. Refactor pipeline thành một entrypoint callable không interactive để CLI, web demo và evaluator dùng lại.
3. Sửa `tests/test_pipeline_utils.py` để import đúng từ `src`.
4. Implement evaluation framework tối thiểu theo `docs/evaluation_metrics_plan.md`.
5. Tạo benchmark `jsonl` tối thiểu và fixture nhỏ để chạy evaluator không cần LLM judge.
6. Đo baseline retrieval trước:
   - `Hit@5`
   - `Recall@5/10/20`
   - `Precision@5`
   - `nDCG@10`
7. Sau đó mới thêm:
   - citation metrics
   - abstention metrics
   - LLM-as-judge cho faithfulness và answer relevance

## Cách khởi động lại trong session mới
Đọc lần lượt:
1. `AGENTS.md`
2. `README.md`
3. `docs/review_fix_plan.md`
4. `docs/evaluation_metrics_plan.md`
5. `docs/mvp_plan.md`
6. `docs/session_handoff.md`

Prompt bootstrap đề xuất:

```text
Hãy đọc AGENTS.md, README.md, docs/review_fix_plan.md, docs/evaluation_metrics_plan.md, docs/mvp_plan.md và docs/session_handoff.md, rồi tóm tắt:
1. những gì đã hoàn thành
2. những gì còn dang dở
3. việc nên làm tiếp theo
```

## Cách verify nhanh môi trường
```bash
docker compose up -d
conda activate RAG
python -m unittest tests.test_pipeline_utils
python src/save_database/save_data.py
python src/embedding/create_db.py
python src/main.py
```

## Blocker và rủi ro hiện tại
- Nếu Gemini quota hết thì bước generation có thể fail với `429`.
- Nếu terminal/input bị lỗi UTF-8, query tiếng Việt có thể hỏng dấu và làm NER/retrieval kém đi.
- Nếu chưa build lại embeddings sau khi đổi dữ liệu graph, kết quả retrieval sẽ không phản ánh dữ liệu mới.
- Test hiện tại cần được rà lại import để đồng bộ hoàn toàn với layout `src/`.
- Chưa có web demo và pipeline callable nên phần MVP mới dừng ở mức kế hoạch + đồng bộ tài liệu.
