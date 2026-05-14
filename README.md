# Traffic Law Assistant (Legal RAG)

![App Screenshot](docs/AppScreenshot.png)

An advanced legal document assistant powered by **LightRAG**, localized for Vietnamese law and featuring high-fidelity Knowledge Graph visualization. This project uses **FastAPI** for the backend, **React** for the frontend, and **PostgreSQL (Apache AGE + pgvector)** for graph and vector storage.

## 🚀 Key Features

- **Vietnamese Legal Localization**: Specialized entity extraction for laws (_Điều khoản, Văn bản pháp luật, Cơ quan ban hành_).
- **Vision-Based PDF Parsing**: Uses **Qwen 3 VL** (via OpenRouter) to extract raw legal text from PDFs with absolute fidelity, even for scans.
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
- **LLM/Embeddings**: Gemini 3 Flash and Qwen 3 VL via OpenRouter, plus local Vietnamese legal embeddings with `huyydangg/DEk21_hcmute_embedding`
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
EMBEDDING_MODEL=huyydangg/DEk21_hcmute_embedding
EMBEDDING_DIM=768
EMBEDDING_DEVICE=cuda
```

### Running the Application

Start the full app with Docker:

```bash
docker compose up --build
```

The application will be available at:

- **Main UI**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **Graph Visualization**: `http://localhost:8001/webui`

### Run With Docker Logs

If you want the main UI and backend together while watching logs:

```bash
docker compose up --build frontend backend
```

In another terminal, follow logs:

```bash
docker compose logs -f frontend backend
```

If you want everything in the background:

```bash
docker compose up -d db backend frontend
docker compose logs -f frontend backend
```

If you want the optional LightRAG Web UI too, start it explicitly because it is behind the `rag-ui` profile:

```bash
docker compose --profile rag-ui up -d rag-ui
docker compose logs -f rag-ui
```

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

- **Primary legal embedding model**: **[huyydangg/DEk21_hcmute_embedding](https://huggingface.co/huyydangg/DEk21_hcmute_embedding)**
- **Embedding backend**: `sentence-transformers`
- **Vector dimension**: `768`
- **Device**: `cuda`
- **Query format**: instruction-style prefix via `EMBEDDING_QUERY_INSTRUCTION`

> [!NOTE]
> Switching from the previous embedding space to `huyydangg/DEk21_hcmute_embedding` (768D) requires a full reindex. Existing vector data must not be mixed across embedding spaces.

> [!IMPORTANT]
> Docker GPU execution requires Docker Desktop GPU support plus an NVIDIA driver/runtime on the host. The backend service is configured with `gpus: all`, so you can inspect logs with `docker compose logs -f backend` while the embedding model runs inside the container.
