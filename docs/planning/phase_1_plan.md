Plan Thực Hiện Phase 1
Trạng thái cập nhật (session hiện tại)
- [x] Bước 1: prepare_phase1_dataset.py
- [x] Bước 2: test cho dataset prep
- [x] Bước 3: baseline BiLSTM trên split mới
- [x] Bước 4: train_phobert_ner.py
- [x] Bước 5: eval_phobert_ner.py
- [x] Bước 6: cập nhật src/NER/ner.py
- [x] Bước 7: cập nhật src/legal_qa.py
- [x] Bước 8: chạy test + smoke test end-to-end

Mục tiêu của phase này:
- fine-tune vinai/phobert-base-v2
- giữ label set cũ:
  - O
  - B-ARTICLE
  - I-ARTICLE
- làm sạch dữ liệu
- bổ sung hard negatives synthetic có kiểm soát
- thay inference BiLSTM hiện tại bằng PhoBERT mà không phá pipeline
Nguyên tắc triển khai
- Không mở rộng label set trong phase này
- So sánh công bằng với BiLSTM trên cùng bộ split sạch
- Ưu tiên giảm false positive trên các câu có từ điều
- Không commit checkpoint lớn vào git nếu file quá nặng
Kế hoạch theo file
1. File mới dự kiến thêm
- src/NER/prepare_phase1_dataset.py
  - deduplicate dữ liệu gốc
  - sinh synthetic hard negatives
  - split train/val/test tránh leakage
  - xuất báo cáo thống kê dataset
- src/NER/train_phobert_ner.py
  - fine-tune vinai/phobert-base-v2 cho token classification
- src/NER/eval_phobert_ner.py
  - đánh giá entity-level metrics trên test set
  - so với baseline BiLSTM nếu cần
- tests/test_ner_dataset_prep.py
  - test dedup
  - test split không overlap
  - test synthetic negatives đều có nhãn O
- tests/test_ner_inference.py
  - test infer() với checkpoint/mock tokenizer-model path mới
2. File sẽ sửa ở bước implement
- src/NER/ner.py
  - đổi inference từ BiLSTM sang PhoBERT
  - thêm cache model/tokenizer
  - giữ API load_model(), predict(), infer()
- src/legal_qa.py
  - đổi model_path mặc định từ bilstm_ner.pt sang thư mục checkpoint PhoBERT
- requirements.txt
  - thêm:
    - transformers
    - seqeval
    - accelerate
Workstream 1: Chuẩn hóa dataset
1. Input gốc
- dùng src/NER/ner_data_8000.json
2. Deduplicate
- key dedup:
  - tuple(tokens)
  - tuple(labels)
- giữ một bản duy nhất cho mỗi exact sample
3. Synthetic hard negatives
- thêm negative unique, tất cả label là O
- chia thành các family:
  - có từ điều nhưng không phải citation
  - legal generic query không có Điều <số>
  - có số nhưng không phải article ref
  - có tên luật/nghị định nhưng không có article ref
- target đề xuất:
  - khoảng 400-500 câu negative unique
- guardrails:
  - không trùng exact với dữ liệu gốc
  - không sinh pattern Điều <số>
  - không sinh câu quá máy móc
  - không sinh các citation trá hình
4. Cân bằng dữ liệu
- sau dedup + synthetic negatives, giữ negative ratio vừa phải
- target hợp lý:
  - negative khoảng 10-20%
- không để negative chiếm quá nhiều vì task hiện vẫn chủ yếu là article spotting
5. Split train/val/test
- split sau khi đã dedup và thêm negatives
- tỉ lệ:
  - 80/10/10
- stratify theo:
  - positive vs negative
  - số entity trong câu: 0, 1, 2+
- kiểm tra:
  - không overlap exact text giữa các split
6. Output artifacts
- dataset sạch:
  - src/NER/processed/phase1_train.json
  - src/NER/processed/phase1_val.json
  - src/NER/processed/phase1_test.json
- báo cáo:
  - src/NER/reports/phase1_dataset_report.json
Workstream 2: Fine-tune PhoBERT
1. Model
- vinai/phobert-base-v2
2. Thiết kế training
- task: token classification
- tokenizer/model từ transformers
- label mapping:
  - O: 0
  - B-ARTICLE: 1
  - I-ARTICLE: 2
3. Label alignment
- tokenize theo PhoBERT subword
- chỉ subword đầu tiên của token nhận label gốc
- subword còn lại gán -100
4. Hyperparameters khởi đầu
- learning rate: 2e-5
- batch size: 16
- epochs: 4-5
- weight decay: 0.01
- warmup ratio: 0.1
- max_length: 128
- early stopping theo validation F1
5. Output checkpoint
- src/NER/checkpoints/phobert_article_ner/
Workstream 3: Đánh giá
1. Baseline cần đo
- BiLSTM hiện tại trên test split mới
2. PhoBERT cần đo
- cùng test split đó
3. Metrics
- entity-level precision
- entity-level recall
- entity-level F1
- token-level accuracy để tham khảo
- false positive rate trên negative queries
4. Slice metrics nên báo riêng
- câu có 1 article
- câu có nhiều article
- câu negative có từ điều
- câu legal generic không có article ref
5. Output report
- src/NER/reports/phase1_eval_report.json
Workstream 4: Tích hợp vào pipeline
1. src/NER/ner.py
- thay kiến trúc BiLSTM bằng PhoBERT inference
- giữ output tương thích:
  - tokens
  - predictions
  - entities
- cache model/tokenizer theo process để tránh load lại mỗi query
2. src/legal_qa.py
- cập nhật _default_ner_infer()
- trỏ tới thư mục checkpoint PhoBERT mới
3. Không thay logic retrieval ở phase này
- mục tiêu là cô lập tác động của NER
Kiểm thử sau tích hợp
- python3 -m unittest tests.test_legal_qa
- python3 -m unittest tests.test_pipeline_utils
- python3 -m unittest tests.test_ner_dataset_prep
- python3 -m unittest tests.test_ner_inference
- smoke test:
  - python3 src/main.py
Acceptance criteria
- dataset split không leak
- synthetic negatives không chứa article citation thật
- PhoBERT vượt BiLSTM về entity F1 trên test set
- false positive trên negative queries giảm
- run_legal_qa() vẫn hoạt động với interface cũ
Thứ tự implement đề xuất
1. prepare_phase1_dataset.py
2. test cho dataset prep
3. baseline BiLSTM trên split mới
4. train_phobert_ner.py
5. eval_phobert_ner.py
6. cập nhật src/NER/ner.py
7. cập nhật src/legal_qa.py
8. chạy test + smoke test
Rủi ro chính cần kiểm soát
- synthetic negatives quá nhiều làm giảm recall
- một số câu có Luật/Nghị định nhưng label vẫn là O, điều này đúng cho phase 1 nhưng sẽ giới hạn năng lực model
- metric token-level có thể đẹp nhưng entity-level chưa chắc tốt, nên phải ưu tiên entity F1
Kết quả mong đợi của phase này
- NER bớt nhầm các câu đời thường có chữ điều
- article extraction tốt hơn BiLSTM
- latency warm inference tốt hơn hiện trạng nếu cache model đúng
