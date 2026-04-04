Scope 1 Implementation Plan
Mục tiêu
- Bổ sung nhận diện document reference cho:
  - Luật <tên>
  - Bộ luật <tên>
  - Nghị định <mã>
- Giữ nguyên:
  - label set NER hiện tại
  - contract run_legal_qa()
  - flow NER -> retrieval -> graph rerank -> Gemini
Nguyên tắc triển khai
- Model NER hiện tại tiếp tục bắt Điều <số>.
- Parser mới chỉ bắt các cụm rất chắc chắn, span ngắn.
- Không cố bắt mọi tên dài ở vòng đầu.
- Ưu tiên ít false positive hơn là coverage rộng.
Phạm vi entity parser ở vòng đầu
- Luật Hải quan
- Luật Doanh nghiệp
- Luật Lao động
- Bộ luật Dân sự
- Bộ luật Lao động
- Nghị định 100/2019/NĐ-CP
- Các mã Nghị định <số>/<năm>/NĐ-CP
Không làm ở vòng này
- Thông tư
- Nghị quyết
- alias phức tạp
- canonicalization sâu theo graph label
- retrain PhoBERT
---
1. File-level plan
src/NER/legal_ref_parser.py
Tạo file mới, chứa logic parser.
Các hàm nên có:
- normalize_whitespace(text: str) -> str
- normalize_entity_text(text: str) -> str
- extract_decree_entities(query: str) -> list[str]
- extract_law_entities(query: str) -> list[str]
- extract_legal_document_entities(query: str) -> list[str]
- merge_legal_entities(base_entities: list[str], parsed_entities: list[str]) -> list[str]
Behavior mong muốn:
- Nghị định 100/2019/NĐ-CP bắt bằng regex chính xác.
- Luật <tên> và Bộ luật <tên> bắt bằng phrase parser ngắn:
  - tối đa 2-4 token tên phía sau
  - dừng khi gặp verb/question marker hoặc dấu câu
- Dedupe theo normalized form.
Guardrails:
- Không bắt:
  - luật sư
  - bộ phận
  - điều kiện
- Không để span ăn sang:
  - quy định gì
  - là gì
  - như thế nào
src/legal_qa.py
Điểm chạm:
- _default_ner_infer()
Kế hoạch:
- giữ nguyên call sang ner.infer(...)
- sau đó merge thêm parser output
- return entity list đã hợp nhất
Kết quả mong muốn:
- query Điều 33 của Luật Hải quan -> ["Điều 33", "Luật Hải quan"]
- query Nghị định 100/2019/NĐ-CP quy định gì -> ["Nghị định 100/2019/NĐ-CP"]
tests/test_ner_inference.py
Bổ sung test parser-focused, không phụ thuộc model thật.
Test positive:
- Luật Hải quan quy định gì? -> có Luật Hải quan
- Bộ luật Dân sự quy định thế nào? -> có Bộ luật Dân sự
- Theo Nghị định 100/2019/NĐ-CP thì sao? -> có Nghị định 100/2019/NĐ-CP
- Điều 33 của Luật Hải quan -> merge được cả article và document
Test normalization:
- luật hải quan -> Luật Hải quan
- nghị định 100/2019/nđ-cp -> Nghị định 100/2019/NĐ-CP
Test negative:
- Tôi muốn hỏi luật sư
- Bộ phận tiếp nhận hồ sơ
- Điều kiện đăng ký
tests/test_legal_qa.py
Bổ sung integration tests nhẹ.
Cases:
- document-only query vẫn tạo ner_entities hữu ích
- mixed query giữ cả article + document
- dedupe khi parser/model cùng trả gần giống nhau
- output contract không đổi
evaluation/fixtures/mini_eval.jsonl
Chưa bắt buộc trong vòng code đầu, nhưng nên cập nhật sau đó.
Nên thêm record:
- Bộ luật Dân sự quy định gì
- Nghị định 100/2019/NĐ-CP quy định gì
- Điều 1 của Luật Hải quan
---
2. Parser design chi tiết
Nghị định <mã>
- Dùng regex chính xác.
- Pattern mục tiêu:
  - Nghị định 100/2019/NĐ-CP
- Output chuẩn:
  - Nghị định 100/2019/NĐ-CP
Luật <tên>
- Chỉ bắt span ngắn, chắc chắn.
- Heuristic:
  - bắt đầu bằng Luật
  - lấy tiếp 1-3 token title-like phía sau
  - dừng nếu gặp:
    - quy định
    - là
    - nói
    - thế
    - ra
    - gì
    - ?
    - ,
- Ví dụ bắt:
  - Luật Hải quan
  - Luật Lao động
- Ví dụ chưa cố bắt dài:
  - Luật Xử lý vi phạm hành chính có thể để vòng sau nếu parser ngắn chưa đủ an toàn
Bộ luật <tên>
- Tương tự Luật, nhưng anchor là Bộ luật
- Ưu tiên:
  - Bộ luật Dân sự
  - Bộ luật Lao động
Merge strategy
- Giữ thứ tự:
  - article entities trước
  - document entities sau
- Dedupe case-insensitive
- Không sort lại phức tạp
---
3. Acceptance criteria vòng này
- Luật Hải quan được bắt đúng
- Bộ luật Dân sự được bắt đúng
- Nghị định 100/2019/NĐ-CP được bắt đúng
- Điều 33 của Luật Hải quan trả ra cả 2 entity
- Không tạo false positive ở các câu negative rõ ràng
- Không làm hỏng test hiện tại
- Không đổi output contract pipeline
---
4. Thứ tự thực hiện
1. Tạo parser file mới
2. Viết unit tests cho parser
3. Hook parser vào src/legal_qa.py
4. Viết integration tests ở tests/test_legal_qa.py
5. Chạy:
   - python3 -m unittest tests.test_ner_inference
   - python3 -m unittest tests.test_legal_qa
   - python3 -m py_compile src/legal_qa.py src/NER/legal_ref_parser.py
---
5. Rủi ro còn lại
- Luật Xử lý vi phạm hành chính là tên dài; với chiến lược span ngắn có thể chưa bắt hết ở vòng đầu.
- Bộ luật hiện chưa chắc có coverage graph tốt như Luật/Nghị định.
- Graph labels hiện không canonical hoàn toàn, nên parser nên normalize vừa phải.
Khuyến nghị chốt để implement
Tôi sẽ implement với coverage ưu tiên:
1. Luật Hải quan
2. Bộ luật Dân sự
3. Nghị định <mã chuẩn>
4. mixed query với Điều <số>