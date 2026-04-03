# Báo cáo so sánh NER Phase 1: PhoBERT vs BiLSTM

## 1) Mục tiêu
- So sánh chất lượng NER giữa mô hình mới `vinai/phobert-base-v2` và baseline `BiLSTM` trên cùng bộ test `src/NER/processed/phase1_test.json`.
- Label set giữ nguyên trong Phase 1: `O`, `B-ARTICLE`, `I-ARTICLE`.

## 2) Nguồn kết quả
- PhoBERT: `src/NER/reports/phobert_phase1_eval.json`
- BiLSTM: `src/NER/reports/bilstm_phase1_eval.json`

## 3) Kết quả tổng quan (overall)

| Metric | PhoBERT | BiLSTM | Chênh lệch (PhoBERT - BiLSTM) |
|---|---:|---:|---:|
| Entity Precision | 1.0000 | 0.9960 | +0.0040 |
| Entity Recall | 1.0000 | 1.0000 | +0.0000 |
| Entity F1 | 1.0000 | 0.9980 | +0.0020 |
| Token Accuracy | 1.0000 | 0.9990 | +0.0010 |
| False Positive Rate | 0.0000 | 0.0784 | -0.0784 |

Nhận xét nhanh:
- PhoBERT vượt BiLSTM ở tất cả metric tổng quan.
- Điểm cải thiện quan trọng nhất là giảm false positive từ `7.84%` xuống `0%`.

## 4) Kết quả theo lát cắt (slices)

### 4.1 Single article (168 mẫu)
- PhoBERT: F1 = 1.0, FPR = 0.0
- BiLSTM: F1 = 1.0, FPR = 0.0
- Kết luận: hai mô hình ngang nhau ở lát cắt này.

### 4.2 Multi article (330 mẫu)
- PhoBERT: F1 = 1.0, FPR = 0.0
- BiLSTM: F1 = 1.0, FPR = 0.0
- Kết luận: hai mô hình ngang nhau ở lát cắt nhiều điều luật.

### 4.3 Negative with "điều" (10 mẫu)
- PhoBERT: token accuracy = 1.0, FPR = 0.0
- BiLSTM: token accuracy = 0.9588, FPR = 0.4
- Kết luận: PhoBERT giảm nhầm mạnh trên câu có từ "điều" nhưng không phải trích dẫn điều luật.

### 4.4 Legal generic no article (32 mẫu)
- PhoBERT: token accuracy = 1.0, FPR = 0.0
- BiLSTM: token accuracy = 0.9966, FPR = 0.03125
- Kết luận: PhoBERT ổn định hơn trên câu pháp lý tổng quát không có `Điều <số>`.

## 5) Đánh giá theo mục tiêu Phase 1

### Mục tiêu kỹ thuật
- Mô hình mới phải tốt hơn baseline trên `phase1_test`.
- Mô hình mới phải giảm nhầm ở nhóm negative.

### Kết quả
- Đạt mục tiêu: PhoBERT tốt hơn BiLSTM về chất lượng tổng quan.
- Đạt mục tiêu trọng tâm: false positive giảm rõ rệt, đặc biệt ở slice `negative_with_dieu`.

## 6) Lưu ý khi diễn giải
- Các lát cắt negative hiện có cỡ mẫu còn nhỏ (`10` và `32`), nên cần thêm holdout khó hơn để xác nhận độ bền ngoài phân phối hiện tại.
- Kết quả hiện tại phù hợp để tiếp tục bước tích hợp PhoBERT vào pipeline, nhưng vẫn nên theo dõi thêm trên query thực tế.

## 7) Kết luận
- Trong phạm vi benchmark Phase 1 hiện tại, `PhoBERT` là lựa chọn tốt hơn `BiLSTM`.
- Cải thiện quan trọng nhất là giảm false positive, giúp ổn định hành vi NER với câu nhiễu có từ "điều".
