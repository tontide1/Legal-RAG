# NER Phase 1 Augmentation Blueprint

## 1) Scope
- Mục tiêu: tăng chất lượng và độ bền cho NER phase 1 với label set giữ nguyên: `O`, `B-ARTICLE`, `I-ARTICLE`.
- Chỉ augment cho train set: `src/NER/processed/phase1_train.json`.
- Không thay đổi `phase1_val.json` và `phase1_test.json`.

## 2) Output files
- `src/NER/processed/phase1_train_augmented.json`
- `src/NER/reports/phase1_train_augmented_audit.json`
- `src/NER/reports/phase1_train_augmented_manifest.json`

## 3) Target size and ratio
- Baseline train hiện tại: `4397` samples.
- Tăng thêm mục tiêu: `+2700` samples.
- Train augmented mục tiêu: khoảng `7097` samples.
- Tỷ lệ mục tiêu sau augment:
  - Positive: `70-80%`
  - Negative: `20-30%`

## 4) Augmentation families and quotas

### A. Hard negatives (không có article) - 1500 mẫu
1. `negative_with_dieu_non_citation` - 500
2. `negative_legal_generic` - 400
3. `negative_numeric_non_article` - 300
4. `negative_adversarial_near_miss` - 300

### B. Positive paraphrase (giữ span article) - 900 mẫu
1. `positive_single_article_paraphrase` - 500
2. `positive_multi_article_paraphrase` - 400

### C. Positive hard format variants - 300 mẫu
1. `positive_roman_numeral_article` - 120
2. `positive_punctuation_and_case_variants` - 100
3. `positive_article_with_doc_context` - 80

## 5) Labeling rules (strict)

### Negative samples
- Toàn bộ token phải là `O`.
- Không chứa article citation hợp lệ theo regex:
  - `(?i)\bđiều\s+(\d+|[IVXLC]+)\b`

### Positive samples
- Mỗi mention article phải được gán:
  - token `Điều` -> `B-ARTICLE`
  - token số/roman ngay sau -> `I-ARTICLE`
- Token ngoài mention article luôn là `O`.
- Nếu có nhiều article trong câu, mỗi article bắt đầu lại bằng `B-ARTICLE`.

## 6) Generation blueprint by family

### 6.1 negative_with_dieu_non_citation (500)
Intent: câu có chữ "điều" nhưng không phải trích dẫn điều luật.

Template examples:
- `Điều này có hợp lý trong trường hợp {context} không?`
- `Tôi đang băn khoăn điều gì quan trọng nhất về {topic}?`
- `Bạn nghĩ điều đó có cần xác nhận thêm không?`

Slots:
- `context`: nộp hồ sơ trực tuyến, thay đổi nơi cư trú, chậm nộp hồ sơ, cập nhật thông tin cá nhân
- `topic`: xử lý vi phạm hành chính, thủ tục cư trú, thuế cá nhân, bảo hiểm xã hội

### 6.2 negative_legal_generic (400)
Intent: câu pháp lý tổng quát có luật/nghị định nhưng không có article citation.

Template examples:
- `Luật {law_name} quy định gì về {topic}?`
- `Nghị định {decree_id} áp dụng cho trường hợp nào?`
- `Bộ luật {law_name} có nguyên tắc nào cho {topic}?`

Slots:
- `law_name`: Đất đai, Doanh nghiệp, Dân sự, Lao động, Bảo hiểm xã hội
- `decree_id`: 100/2019/NĐ-CP, 15/2020/NĐ-CP, 123/2021/NĐ-CP, 45/2022/NĐ-CP
- `topic`: chuyển nhượng, đăng ký cư trú, xử phạt hành chính, hồ sơ giấy phép

### 6.3 negative_numeric_non_article (300)
Intent: có số nhưng không phải article.

Template examples:
- `Mức phạt {amount} đồng có bắt buộc nộp ngay không?`
- `Thời hạn {days} ngày có được gia hạn không?`
- `Sau {days} ngày chưa nộp thì xử lý thế nào?`

Slots:
- `amount`: 500000, 1000000, 2000000, 5000000, 10000000
- `days`: 7, 10, 15, 30, 45, 60, 90

### 6.4 negative_adversarial_near_miss (300)
Intent: câu rất gần positive để ép model học phân biệt.

Template examples:
- `Điều kiện để {action} là gì?`
- `Điều đó có hiệu lực chưa?`
- `Điều này có cần công chứng không?`
- `Bạn muốn biết điều gì nữa về {topic}?`

Slots:
- `action`: xin giấy phép, đăng ký tạm trú, nộp phạt, khiếu nại quyết định xử phạt
- `topic`: cư trú, thuế, bảo hiểm, xử phạt giao thông

### 6.5 positive_single_article_paraphrase (500)
Intent: giữ 1 article mention, thay paraphrase ngữ cảnh.

Template examples:
- `Điều {n} quy định gì?`
- `Theo Điều {n} thì nội dung thế nào?`
- `Nội dung của Điều {n} là gì?`
- `Điều {n} áp dụng ra sao trong trường hợp này?`

Slots:
- `n`: 1-250 (ưu tiên số có trong dữ liệu pháp lý của bạn)

### 6.6 positive_multi_article_paraphrase (400)
Intent: tăng độ bền với nhiều article trong một câu.

Template examples:
- `Điều {a} và Điều {b} quy định gì?`
- `Theo Điều {a}, Điều {b}, Điều {c} thì xử lý thế nào?`
- `Điều {a} khác gì Điều {b}?`

Slots:
- `a,b,c`: số article khác nhau, không trùng trong cùng câu

### 6.7 positive_roman_numeral_article (120)
Intent: tăng coverage cho dạng số La Mã.

Template examples:
- `Điều {roman} quy định gì?`
- `Theo Điều {roman} thì áp dụng như thế nào?`

Slots:
- `roman`: I, II, III, IV, V, VI, VII, VIII, IX, X

### 6.8 positive_punctuation_and_case_variants (100)
Intent: tăng robustness với format viết khác nhau.

Template examples:
- `điều {n} quy định gì?`
- `Điều {n}, quy định thế nào?`
- `Theo điều {n} thì sao?`

### 6.9 positive_article_with_doc_context (80)
Intent: article + tên văn bản trong cùng câu.

Template examples:
- `Điều {n} của Luật {law_name} quy định gì?`
- `Theo Điều {n} trong Nghị định {decree_id} thì mức phạt thế nào?`

## 7) Quality gates (must pass)
1. No exact duplicate với:
   - `phase1_train.json`
   - generated samples cùng batch
   - `phase1_val.json`
   - `phase1_test.json`
2. Negative samples không match regex article citation.
3. Positive samples phải có tối thiểu 1 span article hợp lệ.
4. Label length phải bằng token length.
5. Không cho phép token rỗng, câu rỗng, hoặc câu quá ngắn vô nghĩa.

## 8) Sampling strategy
- Chọn ngẫu nhiên có seed cố định `42` để reproducible.
- Mỗi family sinh dư 20%, rồi lọc theo quality gates để lấy đúng quota.
- Tránh overfit template:
  - mỗi template không vượt quá 15% quota trong family.

## 9) Audit report schema
`phase1_train_augmented_audit.json` cần có:
- tổng mẫu trước/sau augment
- phân bố positive/negative
- phân bố entity buckets: `0`, `1`, `2+`
- phân bố theo `source` và `template_family`
- duplicate removed count
- leakage check count train-vs-val/test

`phase1_train_augmented_manifest.json` cần có cho mỗi mẫu mới:
- `source`
- `template_family`
- `generation_rule_id`
- `seed`

## 10) Suggested execution order
1. Sinh `hard negatives` (A)
2. Sinh `positive paraphrase` (B)
3. Sinh `positive hard format` (C)
4. Chạy quality gates + dedup + leakage filter
5. Merge vào train -> `phase1_train_augmented.json`
6. Xuất audit + manifest
7. Retrain PhoBERT
8. So sánh lại với report hiện tại

## 11) Acceptance criteria
- Augmented train đạt target size và ratio.
- Không có leakage với `val/test`.
- Chất lượng không giảm trên `phase1_test`:
  - `entity_f1` không giảm
  - `false_positive_rate` giữ thấp hoặc giảm
- Cải thiện rõ trên các lát cắt negative khó.
