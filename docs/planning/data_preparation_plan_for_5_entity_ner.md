Dưới đây là gap analysis + kế hoạch data preparation cụ thể cho 5 loại entity, dựa trên code và dữ liệu hiện tại.
Hiện trạng
- Luật <tên>: có source JSON, có trong graph, có trong processed text, nhưng chưa được annotate là entity.
- Bộ luật <tên>: có trong processed text, nhưng chưa thấy source JSON riêng và chưa vào graph.
- Nghị định <mã>: có source JSON, có trong graph, có trong processed text, nhưng chưa được annotate là entity.
- Thông tư <mã>: mới có raw .doc, chưa có JSON, chưa vào graph, chưa thấy coverage rõ trong processed.
- Nghị quyết <mã>: mới có raw .doc, chưa có JSON, chưa vào graph, chưa thấy coverage rõ trong processed.
Các blocker chính
- Label set NER hiện chỉ cho article:
  - src/NER/ner.py:38
  - src/NER/phobert_ner.py:9
- Validator dataset cũng chỉ chấp nhận O, B-ARTICLE, I-ARTICLE:
  - src/NER/prepare_phase1_dataset.py:24-27
  - src/NER/prepare_phase1_dataset.py:50
- Synthetic dataset hiện đang coi câu có tên luật/văn bản nhưng không có Điều là negative:
  - src/NER/prepare_phase1_dataset.py:247-260
- Graph ingest hiện chỉ nạp 3 bộ Luật và 1 bộ Nghị định:
  - src/save_database/save_data.py:18-23
Kết luận hướng làm
Không nên mở rộng multi-label NER ngay. Nên làm theo 2 track song song:
1. Track A: mở rộng coverage dữ liệu và graph
2. Track B: thêm parser/post-processor cho document entities
Mục tiêu là để các query như:
- Luật Hải quan quy định gì
- Theo Nghị định 100/2019/NĐ-CP
- Bộ luật Dân sự
- Thông tư 18/2023/TT-BTC
- Nghị quyết 24/2012/QH13
đều tạo ra entity strings hữu ích cho retrieval, mà chưa phải retrain NER ngay.
---
Plan theo pha
Pha 1: Data inventory và canonical entity list
Mục tiêu:
- lập danh sách chuẩn các tên/mã văn bản mà hệ thống cần nhận diện
Việc cần làm:
- Tạo bảng inventory cho 5 nhóm:
  - Luật
  - Bộ luật
  - Nghị định
  - Thông tư
  - Nghị quyết
- Với mỗi mục, lưu:
  - entity_type
  - canonical_name
  - short_code
  - source_path
  - has_json
  - in_graph
  - in_processed_text
- Trích ra canonical titles từ:
  - dataset/Luật/**/*.json
  - dataset/Nghị_Định/nghidinh.json
  - raw .doc/.txt trong dataset/Thông_tư, dataset/Nghị_Quyết, dataset/Pháp_Lệnh
- Chuẩn hóa biến thể:
  - Nghị định 100/2019/NĐ-CP
  - 100/2019/NĐ-CP
  - NĐ 100/2019
  - Luật Hải quan
  - Bộ luật Dân sự
Kết quả mong muốn:
- Có một inventory rõ ràng để parser dùng lại về sau.
Pha 2: Bổ sung source structured data cho các loại còn thiếu
Mục tiêu:
- đưa Thông tư và Nghị quyết vào cùng mặt bằng với Luật/Nghị định
Việc cần làm:
- Chuyển raw docs của:
  - dataset/Thông_tư
  - dataset/Nghị_Quyết
  thành JSON có schema tương thích ingest hiện tại
- Nếu có thể, bổ sung cả Bộ luật thành source rõ ràng, không chỉ xuất hiện trong synthetic text
- Sau đó mở rộng FILE_PATHS trong ingest để graph thực sự có những loại văn bản này
Lợi ích:
- parser có entity thì retrieval mới thật sự tìm thấy node tương ứng trong graph
Pha 3: Parser document-entity sau NER
Mục tiêu:
- nhận diện tên luật/văn bản mà không đụng label set article hiện tại
Việc cần làm:
- Thêm parser rule-based cho:
  - Luật <tên>
  - Bộ luật <tên>
  - Nghị định <mã>
  - Thông tư <mã>
  - Nghị quyết <mã>
- Merge output parser với output article NER hiện tại
- Dedupe entity strings
- Giữ nguyên contract list[str]
Điểm móc tốt nhất:
- hook sau ner.infer(...) trong _default_ner_infer() của src/legal_qa.py:106
- hoặc tạo helper riêng trong src/NER/
Parser rules nên ưu tiên:
- regex cho mã văn bản:
  - \d+/\d+/NĐ-CP
  - \d+/\d+/TT-[A-Z-]+
  - \d+/\d+/QH\d+
- phrase-based cho title:
  - Luật Hải quan
  - Bộ luật Dân sự
Pha 4: Sửa data NER processed để không “dạy sai”
Mục tiêu:
- tránh việc dữ liệu huấn luyện hiện tại củng cố rằng document names luôn là O
Hiện tại:
- co-ten-van-ban-khong-co-dieu đang là negative family trong prepare_phase1_dataset.py
Plan:
- Tách riêng 2 khái niệm:
  1. negative đối với article extraction
  2. positive đối với document-name extraction
- Nếu vẫn giữ parser-only trước mắt:
  - chưa cần đổi label set ngay
  - nhưng cần tạo benchmark riêng để đánh giá parser
- Nếu chuẩn bị cho retrain sau này:
  - tạo dataset doc-entity thật, thay vì dùng mẫu đang gán O
Pha 5: Benchmark theo 5 loại entity
Mục tiêu:
- đo được parser/retrieval có cải thiện thật không
Tạo tập query nhỏ, chia theo:
- Luật <tên>
- Bộ luật <tên>
- Nghị định <mã>
- Thông tư <mã>
- Nghị quyết <mã>
- câu mix:
  - Điều 33 của Luật Hải quan
  - Theo Điều 8 Nghị định 100/2019/NĐ-CP
Metric nên có:
- entity extraction precision/recall
- retrieval Hit@5 theo từng loại văn bản
- tỷ lệ query document-name mà không còn rơi về full-query fallback
---
Ưu tiên thực hiện
1. Inventory dữ liệu + canonical entity list  
2. Bổ sung source JSON cho Thông tư, Nghị quyết  
3. Mở rộng ingest graph cho 5 loại entity  
4. Thêm parser document-entity  
5. Tạo benchmark và đo retrieval  
6. Chỉ sau đó mới cân nhắc retrain NER đa nhãn
---
Acceptance criteria
- Query chứa Luật/Bộ luật/Nghị định/Thông tư/Nghị quyết tạo ra được entity string đúng
- Các entity đó tồn tại trong graph để retrieval dùng được
- Không làm giảm khả năng nhận Điều <số>
- Không đổi contract run_legal_qa()
- Có benchmark chứng minh query theo tên văn bản được cải thiện
---
Khuyến nghị thực tế
Bản đầu nên nhắm:
- Luật
- Nghị định
- Bộ luật
vì 3 nhóm này đã có tín hiệu trong processed text và dễ đem lại hiệu quả sớm.
Thông tư và Nghị quyết nên làm ngay sau khi có source JSON/ingest.