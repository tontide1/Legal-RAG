# Project Structure Overview

```
Legal-RAG/
├─ .env                # Environment variables (generated from .env.example)
├─ .gitignore
├─ README.md           # High‑level description, screenshots, setup instructions
├─ AGENTS.md           # Notes for AI agents (metadata)
├─ docker-compose.yml  # Services: db, backend, frontend, rag‑ui
├─ backend/            # FastAPI server
│   ├─ main.py                 # App entry point, CORS, router inclusion
│   ├─ config.py               # Pydantic Settings (DB, LLM, embeddings, taxonomy)
│   ├─ requirements.txt        # Python dependencies
│   ├─ api/
│   │   ├─ routes.py          # /chat, /upload, /documents, /health endpoints
│   │   └─ schemas.py         # Pydantic request/response models
│   ├─ core/
│   │   ├─ rag_engine.py      # LightRAG singleton, storage init, async query/insert
│   │   ├─ document_processor.py      # Extracts text from PDF/TXT via Docling without OCR
│   │   ├─ legal_chunker.py   # Normalizes text, enforces Điều boundaries, embeds breadcrumbs
│   │   └─ llm_services.py    # Embedding functions & OpenRouter LLM wrapper
│   └─ tests/                 # Unit tests (not required at runtime)
│       ├─ test_document_processor.py
│       ├─ test_rag_engine.py
│       └─ test_upload_route.py
├─ data/               # LightRAG working directory (default ./backend/data)
├─ db/                 # Custom Postgres Docker image with pgvector & Apache AGE
├─ docs/               # Screenshots & visual assets referenced in README
├─ frontend/           # React UI built with Vite + Tailwind CSS
│   ├─ index.html
│   ├─ vite.config.ts
│   ├─ postcss.config.js
│   ├─ tailwind.config.js
│   ├─ package.json
│   ├─ src/
│   │   ├─ main.tsx
│   │   ├─ index.css
│   │   ├─ App.tsx               # Layout, health check, document list, comparison mode
│   │   └─ components/
│   │       ├─ ChatInterface.tsx  # Handles streaming SSE, renders messages
│   │       └─ FileUpload.tsx     # Upload form, calls /upload API
│   └─ node_modules/            # Installed npm packages (auto‑generated)
└─ .dockerignore
```

## Key Directories
- **backend/** – Core business logic, API, and RAG integration.
- **frontend/** – Interactive UI, health polling, document management.
- **db/** – Dockerfile for Postgres with required extensions.
- **data/** – Persistent storage for LightRAG indexes and temporary caches.
- **docs/** – Visual documentation; not required for execution.

## Files Worth Reading First
1. `backend/config.py` – where all environment variables are defined.
2. `backend/core/rag_engine.py` – initialization of LightRAG, storage, and query flow.
3. `backend/core/document_processor.py` & `backend/core/legal_chunker.py` – ingestion and normalization pipeline.
4. `backend/api/routes.py` – request handling, streaming logic, comparison mode.
5. `frontend/src/App.tsx` – UI entry point and how it communicates with the backend.
6. `docker-compose.yml` – how services are orchestrated; useful for deployment.

## Files Typically Ignored in Production
- `backend/tests/` – test suite, not needed for runtime.
- `frontend/node_modules/` – dependency cache.
- `docs/` image assets.
- `.git/`, `.dockerignore`, `.gitignore`.

## Potential Technical Debt
- Repeated error‑message formatting across routes and LLM wrapper.
- No structured logger; `print` statements are scattered.
- Embedding selection logic duplicated between `RAGEngine._build_embedding_func` and `llm_services`.
- Hard‑coded SSE JSON schema; could be abstracted.
- No authentication – open API endpoints.

---
*Future AI agents can use this hierarchy map to locate relevant modules quickly and understand where new features or fixes should be placed.*
