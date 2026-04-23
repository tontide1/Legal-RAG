1. Objective
- Hoàn thành Phase 1 NER: fine-tune vinai/phobert-base-v2 với label cũ (O, B-ARTICLE, I-ARTICLE), có dataset sạch + hard negatives + split không leakage, và chuẩn bị train/eval scripts trước khi tích hợp vào pipeline QA.
2. What has been completed
- Đã implement dataset prep đầy đủ trong src/NER/prepare_phase1_dataset.py: dedup, synthetic hard negatives (500 mẫu), split 80/10/10 chống leakage, và xuất report.
- Đã tạo artifacts:
  - src/NER/processed/phase1_dedup.json
  - src/NER/processed/phase1_synthetic_negatives.json
  - src/NER/processed/phase1_train.json
  - src/NER/processed/phase1_val.json
  - src/NER/processed/phase1_test.json
  - src/NER/reports/phase1_dataset_audit.json
  - src/NER/reports/phase1_split_report.json
- Đã implement train script trong src/NER/train_phobert_ner.py (manual subword label alignment cho PhoBERT slow tokenizer, Trainer compatibility mới/cũ, check accelerate rõ ràng).
- Đã implement eval script trong src/NER/eval_phobert_ner.py (manual decode token-level từ subword, overall + slice metrics + FPR).
- Đã thêm tests:
  - tests/test_ner_dataset_prep.py
  - tests/test_ner_inference.py
- Đã cập nhật docs/phase_1_plan.md và đánh dấu trạng thái bước đã xong.
3. Files touched
- src/NER/prepare_phase1_dataset.py
- src/NER/train_phobert_ner.py
- src/NER/eval_phobert_ner.py
- tests/test_ner_dataset_prep.py
- tests/test_ner_inference.py
- src/NER/processed/phase1_dedup.json
- src/NER/processed/phase1_synthetic_negatives.json
- src/NER/processed/phase1_train.json
- src/NER/processed/phase1_val.json
- src/NER/processed/phase1_test.json
- src/NER/reports/phase1_dataset_audit.json
- src/NER/reports/phase1_split_report.json
- docs/phase_1_plan.md
4. Decisions already made
- Giữ label set Phase 1: O/B-ARTICLE/I-ARTICLE (không mở rộng label ở phase này).
- Bổ sung hard negatives synthetic có kiểm soát và cân bằng theo family.
- Split phải không leakage theo text key.
- Hỗ trợ PhoBERT slow tokenizer bằng manual alignment thay vì phụ thuộc word_ids().
5. Open questions
- Bạn muốn train full ngay trên toàn bộ phase1_train.json hay chạy pilot nhỏ trước?
- Có cập nhật requirements.txt ngay bây giờ để thêm accelerate (và các deps liên quan) không?
- Có cần chạy baseline BiLSTM trên phase1_test.json ngay để có mốc so sánh trước PhoBERT không?
6. Exact next 3 steps
- Cài dependency còn thiếu trong env RAG (accelerate>=1.1.0).
- Chạy train PhoBERT full và sinh phobert_phase1_train_report.json.
- Chạy eval PhoBERT trên phase1_test.json và đọc entity_f1 + false_positive_rate (overall + slices).
7. Tests or commands to run next
- source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate RAG && python3 -m pip install "accelerate>=1.1.0"
- source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate RAG && python3 src/NER/train_phobert_ner.py --train src/NER/processed/phase1_train.json --val src/NER/processed/phase1_val.json --output-dir src/NER/checkpoints/phobert_article_ner --report-output src/NER/reports/phobert_phase1_train_report.json
- source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate RAG && python3 src/NER/eval_phobert_ner.py --test src/NER/processed/phase1_test.json --checkpoint-dir src/NER/checkpoints/phobert_article_ner --output src/NER/reports/phobert_phase1_eval.json
- source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate RAG && python3 -m unittest tests.test_ner_dataset_prep tests.test_ner_inference
8. Risks / things to avoid
- Không chạy eval trước khi checkpoint train tồn tại (sẽ lỗi path/repo-id).
- Không commit checkpoint/model lớn vào git.
- Không đánh giá bằng token accuracy đơn thuần; ưu tiên entity-level F1 + FPR trên negative.
- Tránh thay đổi retrieval/rerank trong Phase 1 để giữ đo lường NER tách biệt.