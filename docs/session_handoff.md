# Session Handoff

## Mục tiêu của file
File này lưu trạng thái làm việc mới nhất để session sau có thể đọc nhanh và biết:
- đã làm xong gì
- còn dang dở gì
- nên làm gì tiếp theo
- cần kiểm tra lại bằng cách nào

## Đã hoàn thành trong repo
- Ổn định entrypoint QA:
  - sửa đường dẫn model NER trong `Code/main.py`
  - tách cấu hình Gemini sang `Code/pipeline_utils.py`
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
  - `docker-compose.yml` cho Neo4j local
- Thiết lập test tối thiểu:
  - `tests/test_pipeline_utils.py`
  - smoke check bằng `python -m unittest tests.test_pipeline_utils`

## Trạng thái hiện tại cần nhớ
- Repo hiện không có thay đổi chưa commit theo `git status` tại thời điểm tạo handoff này.
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

## Việc còn mở
- Chưa có evaluation framework chạy thật theo plan trong `docs/evaluation_metrics_plan.md`.
- Chưa có benchmark dataset `evaluation/data/legal_rag_eval_v1.jsonl`.
- Chưa có đo lường retrieval/generation/citation bằng số liệu thực tế.
- Chưa có test end-to-end với Neo4j + Gemini thật.
- Chưa có cải tiến cho NER hoặc parser để nhận tên luật/văn bản ngoài `Điều <số>`.

## Việc nên làm tiếp theo
1. Implement evaluation framework theo `docs/evaluation_metrics_plan.md`.
2. Tạo benchmark `jsonl` tối thiểu và fixture nhỏ để chạy evaluator không cần LLM judge.
3. Refactor pipeline thành một entrypoint callable không interactive để evaluator dùng lại.
4. Đo baseline retrieval trước:
   - `Hit@5`
   - `Recall@5/10/20`
   - `Precision@5`
   - `nDCG@10`
5. Sau đó mới thêm:
   - citation metrics
   - abstention metrics
   - LLM-as-judge cho faithfulness và answer relevance

## Cách khởi động lại trong session mới
Đọc lần lượt:
1. `AGENTS.md`
2. `README.md`
3. `docs/review_fix_plan.md`
4. `docs/evaluation_metrics_plan.md`
5. `docs/session_handoff.md`

Prompt bootstrap đề xuất:

```text
Hãy đọc AGENTS.md, README.md, docs/review_fix_plan.md, docs/evaluation_metrics_plan.md và docs/session_handoff.md, rồi tóm tắt:
1. những gì đã hoàn thành
2. những gì còn dang dở
3. việc nên làm tiếp theo
```

## Cách verify nhanh môi trường
```bash
docker compose up -d
conda activate RAG
python -m unittest tests.test_pipeline_utils
python Code/save_database/save_data.py
python Code/embedding/create_db.py
python Code/main.py
```

## Blocker và rủi ro hiện tại
- Nếu Gemini quota hết thì bước generation có thể fail với `429`.
- Nếu terminal/input bị lỗi UTF-8, query tiếng Việt có thể hỏng dấu và làm NER/retrieval kém đi.
- Nếu chưa build lại embeddings sau khi đổi dữ liệu graph, kết quả retrieval sẽ không phản ánh dữ liệu mới.
