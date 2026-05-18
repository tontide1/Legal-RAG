# AI Context Overview

## Project Purpose
- Provide an **interactive legal assistant** for Vietnamese traffic law using Retrieval‑Augmented Generation (RAG).
- Combine **vector search** (embeddings) and **graph search** (knowledge graph) to give precise, citation‑rich answers.

## High‑Level Workflow
```text
User → Frontend (React) → FastAPI /chat endpoint → RAGEngine (LightRAG)
    ↳ Embedding (Qwen / VietLegalHarrier) → Vector store (pgvector)
    ↳ Graph store (Apache AGE) → LLM (OpenRouter Gemini‑3‑flash‑preview)
```
1. **Document ingestion** – PDF/TXT uploaded → `DocumentProcessor` extracts text (via pymupdf4llm for text layer, or PaddleOCR / Chandra OCR 2 for scans) → `legal_chunker` normalizes text by enforcing legal boundaries and embedding breadcrumbs.
2. **Indexing** – `RAGEngine.ainsert` stores normalized chunks, builds embeddings, updates Postgres‑backed vector and graph stores.
3. **Query** – Frontend sends chat request → `routes.chat` builds system prompt → `RAGEngine.aquery` runs LightRAG retrieval (naive or hybrid) → LLM generates answer → streamed back to UI.

## Important Files & Modules
- `backend/main.py` – FastAPI app entry point, CORS setup, includes router.
- `backend/api/routes.py` – API definitions (`/chat`, `/upload`, `/documents`, `/health`).
- `backend/core/rag_engine.py` – Singleton wrapper around LightRAG, builds embedding function, initializes Postgres pools.
- `backend/core/document_processor.py` – Extracts text from PDF/TXT (supports pymupdf4llm, PaddleOCR, and Chandra OCR 2 fallback).
- `backend/core/legal_chunker.py` – Normalizes legal text for LightRAG ingestion (embeds context breadcrumbs, enforces `Điều` boundaries).
- `backend/core/llm_services.py` – Embedding functions (`QwenEmbeddingFunc`, `VietLegalHarrierEmbeddingFunc`) and LLM wrapper (`deepseek_llm_func`).
- `backend/config.py` – Pydantic settings (DB, OpenRouter key, embedding/back‑end config, entity taxonomy).
- `frontend/src/App.tsx` – UI layout, health check, document list, comparison mode toggle.
- `frontend/src/components/*` – Chat interface, file upload component.
- `docker-compose.yml` – Services: Postgres with AGE/pgvector, backend, frontend, rag‑ui.

## Critical Folders
- `backend/` – Server code, core RAG logic, API.
- `frontend/` – React UI, Tailwind CSS, Vite config.
- `data/` – Working directory for LightRAG files (default `./backend/data`).
- `db/` – Custom Postgres Docker image.
- `docs/` – Screenshots, README assets.

## Entry Points
- `docker compose up -d` → spins up all services.
- `backend/main.py` can also be run directly: `uvicorn backend.main:app`.
- Frontend dev: `npm run dev` inside `frontend/`.

## Coding Conventions
- **Async‑first**: All I/O (file, DB, LLM) is async to keep FastAPI performant.
- **Pydantic Settings** – centralised config, loaded from `.env`.
- **Dependency Injection** – embedding and LLM functions passed to LightRAG via callbacks.
- **Type hints** – everywhere, useful for IDEs.
- **Explicit error handling** – HTTPException on API errors, streaming error messages.

## Current Limitations / Known Issues
- **Embedding backend switch requires full re‑index** (vector dimensions differ).
- **PDF OCR is CPU‑heavy**; large scanned PDFs relying on PaddleOCR or Chandra OCR 2 may cause timeouts.
- **Comparison mode spawns two parallel async tasks** – may double DB load.
- No authentication/authorization – open endpoint (dev only).
- The `EMBEDDING_QUERY_INSTRUCTION` is hard‑coded for Vietnamese law; non‑Vietnamese queries may perform poorly.

## Assumptions
- All documents are in Vietnamese legal domain.
- PostgreSQL is reachable via environment variables.
- OpenRouter API key is valid and has sufficient quota.

## Component Interaction Summary (recommended reading order)
1. **Config (`backend/config.py`)** – understand environment variables.
2. **RAGEngine (`backend/core/rag_engine.py`)** – initialization, storage setup.
3. **DocumentProcessor & Legal Chunker (`backend/core/document_processor.py`, `backend/core/legal_chunker.py`)** – ingestion and normalization pipeline.
4. **LLM Services (`backend/core/llm_services.py`)** – embeddings & LLM wrapper.
5. **API Routes (`backend/api/routes.py`)** – request flow.
6. **Frontend (`frontend/src/App.tsx` & components)** – UI triggers.

## Files Safe to Ignore / Experimental
- `backend/tests/` – unit tests, not required for production run.
- `.dockerignore`, `.gitignore` – tooling.
- `AGENTS.md` – meta‑doc for other agents.
- `docs/` images – only for documentation.
- `backend/__pycache__/` – compiled bytecode.

## Duplicated Logic / Technical Debt
- **Embedding selection** duplicated between `RAGEngine._build_embedding_func` and `llm_services` – could be unified.
- **Error message formatting** in routes and LLM wrapper repeats similar string building.
- **Hard‑coded streaming JSON schema** (`type`, `mode`, `content`) – could be abstracted to a helper.
- No centralized logger; `print` statements scatter throughout code.

---
*Future AI agents should start by reading this file, then dive into `backend/config.py` and `backend/core/rag_engine.py` for core behavior.*
