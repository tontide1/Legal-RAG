# Kế hoạch MVP Legal Graph RAG cho đồ án đại học

## Tóm tắt
Mục tiêu là hoàn thiện một MVP web demo trong 1 tuần cho chatbot pháp luật tiếng Việt, dựa trên codebase Graph RAG hiện tại. Hệ thống sẽ được định vị là **Vietnamese Legal Graph RAG with NAGphormer-inspired Graph Embeddings**: có pipeline end-to-end, có Neo4j, hybrid retrieval, graph rerank, nhưng không tuyên bố đã tái hiện đầy đủ bài nghiên cứu. Trọng tâm là làm cho sản phẩm chạy ổn, câu trả lời có căn cứ, có web demo để trình bày, và có benchmark nhỏ để báo cáo thuyết phục.

## Đánh giá codebase hiện tại so với bài nghiên cứu

### Điểm mạnh
- Đã có pipeline end-to-end: nhập dữ liệu, tạo embedding, retrieval, generation.
- Đã dùng Neo4j làm graph backend thay vì chỉ text/vector retrieval.
- Đã có hybrid retrieval giữa BM25 và SBERT.
- Đã có graph rerank sử dụng graph embedding.
- Đã có phiên bản NAGphormer rút gọn trong phần graph embedding.
- Dữ liệu nguồn đã phản ánh phần nào cấu trúc pháp lý như văn bản, chương, mục, điều.
- Prompt và luồng hỏi đáp đang đi theo đúng bài toán pháp luật tiếng Việt.

### Điểm yếu
- Khoảng cách giữa research paper và code thật còn lớn.
- Quan hệ đồ thị hiện chủ yếu là quan hệ cấu trúc kiểu `bao_gồm`, chưa có nhiều legal relations giàu ngữ nghĩa như:
  - `hướng_dẫn_thi_hành`
  - `sửa_đổi`
  - `bãi_bỏ`
  - `thay_thế`
  - `có_hiệu_lực_từ`
  - `hết_hiệu_lực`
- NAGphormer hiện là prototype đơn giản:
  - dùng one-hot node feature
  - train theo reconstruction
  - chưa có temporal modeling
  - chưa có heterogeneous graph modeling
  - chưa có legal ontology reasoning
- NER hiện chỉ mạnh với dạng `Điều <số>`, chưa nhận tốt tên luật/văn bản.
- Chưa có evaluator thật để đo retrieval/generation bằng số liệu.
- Chưa có web demo phục vụ trình bày.
- Có drift kỹ thuật trong repo như tài liệu cũ nhắc `Code/` trong khi code hiện ở `src/`.

### Kết luận đánh giá
Codebase hiện tại phù hợp để phát triển thành **MVP đồ án đại học có tính ứng dụng**, nhưng chưa đủ để tự nhận là hệ thống hiện thực đầy đủ kiến trúc research-level theo paper. Hướng đúng là:
- giữ NAGphormer như lõi ý tưởng kỹ thuật ở mức prototype,
- tối ưu độ đúng và tính ổn định của chatbot,
- bổ sung demo web và benchmark nhỏ,
- trình bày trung thực đâu là phần đã hiện thực, đâu là phần hướng phát triển.

## Mục tiêu MVP
MVP phải đạt được 4 mục tiêu:
- Có web demo chạy ổn định để hội đồng có thể dùng thử.
- Trả lời được câu hỏi pháp luật tiếng Việt dựa trên căn cứ retrieve từ graph.
- Hiển thị được căn cứ pháp lý và luồng xử lý cơ bản.
- Có số liệu benchmark tối thiểu để đưa vào báo cáo.

## Phạm vi MVP

### Trong phạm vi
- Neo4j local.
- Dữ liệu hiện có trong `dataset/`.
- Graph RAG với:
  - NER hiện tại
  - BM25
  - SBERT
  - graph rerank
  - Gemini answer generation
- Web demo bằng Streamlit.
- Benchmark nhỏ cho báo cáo.
- Citation hiển thị từ retrieved nodes.
- Fallback an toàn khi không đủ căn cứ.

### Ngoài phạm vi
- Reproduce đầy đủ paper NAGphormer.
- Agentic multi-agent system.
- Temporal legal knowledge graph hoàn chỉnh.
- Mở rộng NER thành bộ nhận dạng pháp lý tổng quát.
- Xây ontology pháp lý đầy đủ.
- Phát triển frontend lớn kiểu production.
- Hệ thống triển khai cloud thật.

## Kiến trúc MVP đề xuất

### 1. Refactor pipeline thành API callable
Tạo một hàm dùng chung, ví dụ:
`run_legal_qa(query: str) -> dict`

Output chuẩn nên gồm:
- `query`
- `ner_entities`
- `retrieved_nodes`
- `context_text`
- `answer_text`
- `citations`
- `scores`
- `timings`
- `errors`

Hàm này là lõi dùng lại cho:
- CLI
- Streamlit app
- evaluator

### 2. Ổn định retrieval trước generation
Không mở rộng thuật toán lớn trong 1 tuần. Tập trung:
- chuẩn hóa input query,
- fallback khi NER không nhận được entity,
- chọn top `3-5` căn cứ tốt nhất,
- không nhồi quá nhiều context vào prompt,
- chỉ generate khi retrieval đủ mạnh,
- nếu retrieval yếu thì trả lời kiểu:
  `Tôi không tìm thấy căn cứ pháp lý đủ rõ cho trường hợp này`.

### 3. Citation và explainability
Không trích citation tự do từ LLM. Citation của MVP phải xuất phát từ retrieved nodes:
- tên văn bản
- điều luật
- label
- score

Web demo phải hiển thị:
- thực thể NER đã nhận diện
- top retrieved nodes
- answer cuối cùng
- căn cứ pháp lý đi kèm

### 4. Web demo bằng Streamlit
Giao diện tối thiểu gồm 3 phần:
- ô nhập câu hỏi pháp luật tiếng Việt
- bảng/top list căn cứ retrieve được
- câu trả lời cuối cùng của chatbot

Phần phụ:
- trạng thái môi trường
- thời gian xử lý
- mô tả ngắn pipeline

### 5. Benchmark nhỏ cho báo cáo
Tạo benchmark nhỏ `30-50` câu hỏi, chia theo nhóm:
- hỏi trực tiếp theo điều luật
- hỏi theo tên luật/văn bản
- hỏi theo nội dung/chủ đề
- câu không đủ căn cứ để trả lời

Metric tối thiểu:
- `Hit@5`
- `Recall@5`
- `citation_precision`
- tỷ lệ abstain đúng với câu không trả lời được

## Kế hoạch triển khai trong 7 ngày

### Ngày 1
- Refactor pipeline thành hàm callable dùng chung.
- Sửa các import/path drift trong repo.
- Đảm bảo luồng `save_data -> create_db -> query` chạy end-to-end.

### Ngày 2
- Chuẩn hóa retrieval output.
- Thêm citation extraction từ retrieved nodes.
- Thêm fallback logic khi retrieval yếu.
- Chuẩn hóa format answer để ổn định demo.

### Ngày 3
- Xây Streamlit demo.
- Nối web UI với pipeline callable.
- Hiển thị top căn cứ, answer, timings, errors.

### Ngày 4
- Tạo benchmark nhỏ bằng tay.
- Viết evaluator tối thiểu cho retrieval và citation.
- Chạy thử trên dữ liệu nhỏ.

### Ngày 5
- Tinh chỉnh prompt answer generation.
- Chạy baseline và thu số liệu.
- Chốt các ví dụ demo đẹp để trình bày.

### Ngày 6
- Làm sạch UX của demo.
- Chụp ảnh màn hình.
- Chuẩn bị sơ đồ pipeline và bảng so sánh paper vs MVP.

### Ngày 7
- Smoke test cuối.
- Chốt số liệu trong báo cáo.
- Hoàn thiện phần trình bày và kết luận.

## Test plan

### Smoke tests
- Import dữ liệu vào Neo4j thành công.
- Tạo content embedding và graph embedding thành công.
- Query mẫu trả về top retrieved nodes và answer.
- Streamlit app chạy được local.

### Regression tests
- Test `pipeline_utils`.
- Test formatting output của pipeline callable.
- Test citation extraction.
- Test fallback khi không có NER entity.
- Test trường hợp không có kết quả retrieval.

### Acceptance criteria
- Người dùng nhập câu hỏi tiếng Việt trên web và nhận được câu trả lời.
- Hệ thống hiển thị được căn cứ pháp lý cụ thể.
- Có benchmark nhỏ và số liệu thật để đưa vào báo cáo.
- Báo cáo giải thích rõ quan hệ giữa paper và MVP hiện thực.

## Cách định vị học thuật trong báo cáo
Tên đề tài hoặc định vị nên dùng:
**Vietnamese Legal Graph RAG with NAGphormer-inspired Graph Embeddings**

Cần trình bày rõ:
- Hệ thống có hiện thực Graph RAG thực tế.
- Có sử dụng graph embedding theo tinh thần NAGphormer.
- Chưa hiện thực đầy đủ:
  - temporal graph
  - legal ontology hoàn chỉnh
  - agentic routing
  - heterogeneous legal knowledge graph chuẩn research paper

Cách nói này vừa trung thực, vừa vẫn giữ được giá trị học thuật.

## Assumptions
- Thời gian hoàn thiện còn đúng khoảng 1 tuần.
- Web demo dùng Streamlit.
- Mục tiêu ưu tiên là sản phẩm chạy ổn và có độ chính xác đầu ra tốt hơn.
- NAGphormer được giữ làm lõi kỹ thuật ở mức prototype khả thi.
- Không mở rộng phạm vi sang production system hoặc full research reproduction.
