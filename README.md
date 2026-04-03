# Legal-RAG

Pipeline hỏi đáp pháp luật tiếng Việt dùng Neo4j + NER + hybrid retrieval + Gemini.

## Thành phần chính
- `src/main.py`: CLI hỏi đáp pháp luật
- `src/save_database/save_data.py`: nạp dữ liệu luật vào Neo4j
- `src/embedding/create_db.py`: tạo content embedding và graph embedding
- `src/retrive/multi_retr.py`: retrieval BM25 + SBERT + graph rerank
- `src/NER/ner.py`: BiLSTM NER cho span `Điều <số>`
- `docker-compose.yml`: Neo4j local cho môi trường dev

## Tài liệu chính
- `docs/review_fix_plan.md`: tổng hợp các sửa lỗi kỹ thuật đã chốt
- `docs/evaluation_metrics_plan.md`: kế hoạch đo lường retrieval/generation
- `docs/mvp_plan.md`: kế hoạch MVP phù hợp cho demo và báo cáo đồ án
- `docs/session_handoff.md`: trạng thái làm việc và việc nên làm tiếp theo

## Agent skills (Codex + OpenCode)
- Skill hiện được mirror song song tại:
  - `.codex/skills/<skill-name>/SKILL.md`
  - `.opencode/skills/<skill-name>/SKILL.md`
- Nguồn chuẩn là `.codex/skills`; `.opencode/skills` phải đồng bộ 1-1.
- Kiểm tra nhanh:
```bash
python3 scripts/validate_skills.py --repo-root .
python3 -m unittest tests.test_skill_validation
```

## Yêu cầu
- Python 3.11+
- Conda environment `RAG`
- Docker để chạy Neo4j local

## Cài đặt
```bash
conda create -n RAG python=3.11 -y
conda activate RAG
python -m pip install --upgrade pip
python -m pip install \
  torch numpy python-dotenv neo4j rank-bm25 sentence-transformers \
  langchain langchain-core langchain-google-genai huggingface-hub \
  torch-geometric
```

## Cấu hình `.env`
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-2.5-flash-lite
HUGGINGFACEHUB_API_TOKEN=optional
```

## Chạy Neo4j
```bash
docker compose up -d
```

Neo4j UI:
- `http://localhost:7474`
- user: `neo4j`
- password: lấy từ `NEO4J_PASSWORD`

## Chạy project
```bash
conda activate RAG
python src/save_database/save_data.py
python src/embedding/create_db.py
python src/main.py
```

## Chạy giao diện Streamlit
```bash
conda activate RAG
streamlit run streamlit_app.py
```

Giao diện web sẽ mở tại `http://localhost:8501`.

## Test nhanh
```bash
conda activate RAG
python -m unittest tests.test_pipeline_utils
python -m py_compile src/main.py src/save_database/save_data.py src/embedding/create_db.py
```

## Ghi chú kỹ thuật
- Neo4j app nodes hiện dùng label `LegalRAG`
- Định danh node dùng `node_id = "{Label}::{Tên}"`
- Gemini model mặc định hiện tại là `gemini-2.5-flash-lite`
- NER hiện chỉ mạnh với span tham chiếu điều luật như `Điều 33`, chưa phải NER tổng quát cho mọi tên luật/văn bản
- Nếu gặp lỗi `429 RESOURCE_EXHAUSTED`, nguyên nhân nằm ở quota/rate limit của Gemini API key
- MVP hiện tại được định vị là legal Graph RAG lấy cảm hứng từ NAGphormer, chưa phải bản tái hiện đầy đủ bài nghiên cứu

## Lỗi thường gặp
- **Thiếu `GOOGLE_API_KEY`**: Gemini không thể generate câu trả lời. Kiểm tra file `.env`.
- **Thiếu `NEO4J_PASSWORD`**: Không thể kết nối Neo4j để retrieval. Kiểm tra file `.env` và đảm bảo Docker đang chạy.
- **`429 RESOURCE_EXHAUSTED`**: Gemini API key đã vượt quota/rate limit. Đợi hoặc đổi API key khác.
