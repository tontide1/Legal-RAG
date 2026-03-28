# Legal-RAG

Pipeline hỏi đáp pháp luật tiếng Việt dùng Neo4j + NER + hybrid retrieval + Gemini.

## Thành phần chính
- `Code/main.py`: CLI hỏi đáp pháp luật
- `Code/save_database/save_data.py`: nạp dữ liệu luật vào Neo4j
- `Code/embedding/create_db.py`: tạo content embedding và graph embedding
- `Code/retrive/multi_retr.py`: retrieval BM25 + SBERT + graph rerank
- `Code/NER/ner.py`: BiLSTM NER cho span `Điều <số>`
- `docker-compose.yml`: Neo4j local cho môi trường dev

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
python Code/save_database/save_data.py
python Code/embedding/create_db.py
python Code/main.py
```

## Test nhanh
```bash
conda activate RAG
python -m unittest tests.test_pipeline_utils
python -m py_compile Code/main.py Code/save_database/save_data.py Code/embedding/create_db.py
```

## Ghi chú kỹ thuật
- Neo4j app nodes hiện dùng label `LegalRAG`
- Định danh node dùng `node_id = "{Label}::{Tên}"`
- Gemini model mặc định hiện tại là `gemini-2.5-flash-lite`
- NER hiện chỉ mạnh với span tham chiếu điều luật như `Điều 33`, chưa phải NER tổng quát cho mọi tên luật/văn bản
- Nếu gặp lỗi `429 RESOURCE_EXHAUSTED`, nguyên nhân nằm ở quota/rate limit của Gemini API key
