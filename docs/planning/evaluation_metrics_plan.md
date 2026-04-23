# Kế hoạch Metrics và Implementation cho Graph RAG Legal Vietnamese

## Tóm tắt
Thiết lập một bộ đánh giá MVP nhưng đủ số liệu thật cho toàn pipeline `NER -> retrieval -> graph rerank -> answer generation`, theo hướng hybrid: nhãn tay cho retrieval/citation/core correctness, LLM-as-judge cho faithfulness và relevance. Bộ metric dựa trên IR kiểu BEIR/TREC Legal cho retrieval và Ragas/TruLens-style cho answer quality.

Tài liệu này là kế hoạch evaluation chuyên sâu. Phạm vi MVP tổng thể của đồ án được mô tả riêng tại `docs/mvp_plan.md`.

## Thiết kế benchmark và metric
- Tạo bộ benchmark `evaluation/data/legal_rag_eval_v1.jsonl` với `120` query tiếng Việt.
- Chia query thành 6 nhóm:
  - `25` hỏi trực tiếp `Điều <số>`
  - `20` hỏi theo tên luật/văn bản
  - `20` hỏi theo chủ đề/nội dung điều luật
  - `20` cần nhiều căn cứ hoặc nhiều điều luật
  - `15` có metadata như ngày ban hành/số hiệu
  - `20` câu không trả lời được từ KB hoặc dễ gây hallucination
- Mỗi record phải có các trường:
  - `id`, `query`, `category`, `answerable`
  - `gold_ner_entities`
  - `gold_node_ids`
  - `gold_citations`
  - `reference_answer`
  - `notes`
- Metric bắt buộc:
  - NER: strict `precision`, `recall`, `f1` trên `gold_ner_entities`
  - Retrieval: `Hit@5`, `Recall@5/10/20`, `Precision@5`, `nDCG@10`
  - Graph rerank: `delta Recall@10` và `delta nDCG@10` giữa trước và sau rerank
  - Generation: `faithfulness`, `answer_relevance`, `answer_correctness`
  - Legal citation: `citation_precision`, `citation_recall`, `citation_exact_match`
  - Safety: `abstention_precision` cho query `answerable=false`
  - Ops: `latency_p50`, `latency_p95`, `retrieval_latency`, `generation_latency`, `error_rate`, `quota_error_rate`
- Quy ước chấm retrieval:
  - Binary relevance cho MVP: node thuộc `gold_node_ids` là relevant, còn lại là non-relevant
  - Không dùng graded relevance ở vòng đầu để giảm chi phí annotate
- Quy ước chấm citation:
  - Chuẩn hóa citation theo regex/heuristic cho `Điều`, `Luật`, `Nghị định`
  - So sánh với `gold_citations`
- Metric LLM-as-judge:
  - `faithfulness`: answer có bám đúng retrieved context hay không
  - `answer_relevance`: answer có trả đúng câu hỏi hay không
  - Dùng judge model tách riêng khỏi answer generation qua biến `EVAL_JUDGE_MODEL`

## Kế hoạch implementation
- Thêm thư mục `evaluation/` với cấu trúc:
  - `evaluation/data/legal_rag_eval_v1.jsonl`
  - `evaluation/run_eval.py`
  - `evaluation/metrics.py`
  - `evaluation/judge.py`
  - `evaluation/report.py`
  - `evaluation/fixtures/mini_eval.jsonl`
- Refactor pipeline để có entrypoint không tương tác, ví dụ một hàm dùng lại được từ CLI và evaluator:
  - Input: `query`, cờ bật/tắt LLM judge
  - Output chuẩn hóa: `ner_entities`, `retrieved_nodes`, `answer_text`, `raw_context`, `citations_found`, `timings`, `errors`
- Trong evaluator:
  - Bước 1: chạy toàn bộ benchmark qua pipeline
  - Bước 2: chấm NER và retrieval bằng metric deterministic
  - Bước 3: chấm citation bằng parser deterministic
  - Bước 4: chấm `faithfulness` và `answer_relevance` bằng judge model
  - Bước 5: tổng hợp `summary.json` và `summary.md`
- Thêm biến môi trường:
  - `EVAL_JUDGE_MODEL` mặc định `gemini-2.5-flash-lite`
  - `EVAL_ENABLE_LLM_JUDGE` mặc định `true`
- Hành vi khi judge bị quota/429:
  - Không fail toàn bộ evaluation
  - Vẫn xuất retrieval/NER/ops metrics
  - Gắn trạng thái `judge_metrics_skipped=true` trong report
- Thêm ablation mode trong evaluator:
  - `bm25_only`
  - `semantic_only`
  - `hybrid_no_graph`
  - `hybrid_with_graph`
- Report đầu ra:
  - `evaluation/reports/latest/summary.json`
  - `evaluation/reports/latest/summary.md`
  - `evaluation/reports/latest/per_query.jsonl`

## Test và tiêu chí chấp nhận
- Unit test cho:
  - `nDCG@10`, `Recall@k`, `citation parser`, `citation scoring`, `abstention scoring`
- Fixture test với `mini_eval.jsonl` cỡ `3-5` query để kiểm tra end-to-end evaluator
- Smoke test:
  - chạy evaluator với `EVAL_ENABLE_LLM_JUDGE=false` vẫn ra report hoàn chỉnh
  - chạy evaluator khi gặp `429` phải không crash toàn bộ
- Acceptance criteria cho vòng đầu:
  - Benchmark chạy hết và sinh đủ `summary.json`, `summary.md`, `per_query.jsonl`
  - Có số liệu riêng cho từng category query
  - Có bảng so sánh trước/sau graph rerank
  - Có thể xem top lỗi theo nhóm: NER miss, retrieval miss, citation sai, answer không grounded
- Baseline goals ban đầu để theo dõi:
  - `Recall@10 >= 0.85`
  - `nDCG@10 >= 0.70`
  - `citation_precision >= 0.80`
  - `abstention_precision >= 0.90`
  - `graph rerank lift >= 0` trên `Recall@10` và `nDCG@10`
- Các threshold này là mốc khởi đầu; giữ nguyên trong v1 để có baseline so sánh, chỉ thay sau khi đã có một lần đo chính thức.

## Giả định đã chốt
- Giai đoạn đầu là `MVP nhưng đủ số liệu`
- Cách chấm là `hybrid`: nhãn tay + LLM-as-judge
- Bộ benchmark đầu tiên dùng `120` query
- Retrieval dùng binary relevance cho v1
- Judge model mặc định là `gemini-2.5-flash-lite`
- Không mở rộng NER label set trong plan này; evaluator sẽ đo đúng tình trạng hiện tại của NER
