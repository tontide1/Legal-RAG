# Traffic Law Assistant (Legal RAG)

![App Screenshot](docs/AppScreenshot.png)

An advanced legal document assistant powered by **LightRAG**, localized for Vietnamese law and featuring high-fidelity Knowledge Graph visualization. This project uses **FastAPI** for the backend, **React** for the frontend, and **PostgreSQL (Apache AGE + pgvector)** for graph and vector storage.

## 🚀 Key Features

- **Vietnamese Legal Localization**: Specialized entity extraction for laws (_Điều khoản, Văn bản pháp luật, Cơ quan ban hành_).
- **Local PDF OCR**: Uses `pdf2image` + PaddleOCR to extract raw legal text from PDFs, including scans.
- **Interactive Knowledge Graph**: Explore legal relationships via the integrated **LightRAG Graph UI** on port 8001.

  ![KG Screenshot 1](docs/KGScreenshot1.png)
  ![KG Screenshot 2](docs/KGScreenshot2.png)

- **Comparison Mode**: Side-by-side RAG evaluation with parallel streaming.

  ![Comparison 1](docs/Comparison1.png)
  ![Comparison 2](docs/Comparison2.png)
  ![Comparison 3](docs/Comparison3.png)
  ![Comparison 4](docs/Comparison4.png)
  ![Comparison 5](docs/Comparison5.png)

- **Hybrid RAG Retrieval**: Combined vector and graph search for precise legal grounding.
- **Modern Chat Interface**: Beautiful React UI with Markdown support and source citations.
- **Document Inventory**: Manage and track the status of all indexed legal documents.

## 🛠 Tech Stack

- **Backend**: Python 3.11, FastAPI, `lightrag-hku`
- **Frontend**: Vite, React, TypeScript, Tailwind CSS, Shadcn UI
- **Database**: PostgreSQL with `pgvector` (Vector) and `Apache AGE` (Graph)
- **LLM/Embeddings**: Gemini 3 Flash via OpenRouter, plus local Vietnamese legal embeddings with `mainguyen9/vietlegal-harrier-0.6b`
- **Gemini response models**: `gemini-3-flash-preview` (default), plus optional `gemini-2.5-flash-lite`, `gemini-3-flash` and `gemini-3.1-flash-lite`
- **Deployment**: Docker Compose

## 📦 Getting Started

### Prerequisites

- Docker and Docker Compose
- OpenRouter API Key

### Environment Setup

Create a `.env` file in the root directory (refer to `.env.example`):

```bash
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DATABASE=law_assistant
OPENROUTER_API_KEY=your_key_here
LLM_MODEL=gemini-3-flash-preview
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=mainguyen9/vietlegal-harrier-0.6b
EMBEDDING_DIM=1024
```

### Running the Application

1. **Start the Infrastructure**:

   ```bash
   docker compose up -d
   ```

2. **Start the Frontend (Locally)**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

The application will be available at:

- **Main UI**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000`
- **Graph Visualization**: `http://localhost:8001/webui`

## 🧠 Architecture

The system consists of three main services:

- `db`: Custom Postgres image with vector and graph extensions.
- `backend`: Handles chat, PDF parsing, and document indexing.
- `rag-ui`: Provides the Knowledge Graph visualization interface.

## 🇻🇳 Localization Details

The RAG engine is optimized for Vietnamese:

- `SUMMARY_LANGUAGE`: Set to `Vietnamese`.
- `ENTITY_TYPES`: Custom legal taxonomy including _Hành vi vi phạm, Hình thức xử phạt, Khái niệm pháp lý_.

## 🌍 Embedding Setup

This repository now supports a local Hugging Face embedding backend for Vietnamese legal retrieval:

- **Primary legal embedding model**: **[mainguyen9/vietlegal-harrier-0.6b](https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b)**
- **Embedding backend**: `sentence-transformers`
- **Vector dimension**: `1024`
- **Query format**: instruction-style prefix via `EMBEDDING_QUERY_INSTRUCTION`

> [!NOTE]
> Switching from `openai/text-embedding-3-small` (1536D) to `mainguyen9/vietlegal-harrier-0.6b` (1024D) requires a full reindex. Existing vector data must not be mixed across embedding spaces.
