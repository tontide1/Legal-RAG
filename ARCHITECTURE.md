# Architecture Overview

## High‑Level System Diagram
```mermaid
flowchart TD
    subgraph Frontend[React Frontend]
        UI[User Interface]
        Upload[File Upload Component]
        Chat[Chat Interface]
    end
    subgraph Backend[FastAPI Backend]
        API[API Router]
        RAG[RAGEngine (LightRAG)]
        DocProc[DocumentProcessor]
        Chunker[Legal Chunker]
        LLM[LLM Wrapper (deepseek_llm_func)]
        Emb[Embedding Functions]
        DB[Postgres + pgvector + Apache AGE]
    end
    UI -->|upload file| Upload -->|POST /upload| API
    UI -->|send chat| Chat -->|POST /chat| API
    API --> DocProc -->|extract text| Chunker
    Chunker -->|normalize text| RAG
    API --> RAG
    RAG -->|store chunks| DB
    RAG -->|retrieve (vector/graph)| DB
    RAG -->|call LLM| LLM -->|return answer| API -->|stream response| Chat
    Chat -->|display answer| UI
```

## Core Components
| Component | Responsibility | Key Files |
|-----------|----------------|-----------|
| **FastAPI entry point** | Starts the app, sets CORS, includes routes. | `backend/main.py` |
| **API router** | Defines `/chat`, `/upload`, `/documents`, `/health`. Handles request validation, streaming logic, error handling. | `backend/api/routes.py` |
| **RAGEngine** | Singleton wrapper around **LightRAG**. Initializes Postgres connection pools, builds embedding function, provides async `aquery` and `ainsert`. | `backend/core/rag_engine.py` |
| **DocumentProcessor** | Converts PDF or plain TXT to text. Uses Docling for PDFs with OCR disabled. Writes extracted PDF text to `extracted_txt/<stem>.txt` and rejects scanned/image-only PDFs with no embedded text. Supports async extraction. | `backend/core/document_processor.py` |
| **Legal Chunker** | Normalizes legal text by embedding context breadcrumbs (`[Chương > Mục]`) and enforcing `Điều` boundaries for optimal LightRAG chunking. | `backend/core/legal_chunker.py` |
| **Embedding Functions** | - `QwenEmbeddingFunc` – uses OpenRouter embeddings (via `openai.AsyncOpenAI`).<br>- `VietLegalHarrierEmbeddingFunc` – local `sentence‑transformers` model with a query‑instruction prefix. | `backend/core/llm_services.py` |
| **LLM Wrapper** | Calls OpenRouter LLM (`gemini-3-flash-preview` by default). Handles streaming and parameter sanitisation. | `backend/core/llm_services.py` |
| **Postgres Storage** | Vector store (`pgvector`), graph store (`Apache AGE`), KV and doc‑status tables, all accessed through LightRAG adapters. | Configured in `backend/core/rag_engine.py` (environment variables). |
| **Frontend** | React app built with Vite + Tailwind. Provides health check, document list, comparison mode (parallel naive vs hybrid streams). | `frontend/src/App.tsx`, `frontend/src/components/*` |

## Data Flow
1. **Ingestion**
   - User uploads a PDF/TXT via `/upload`.
   - `DocumentProcessor.extract_text` → extracts embedded PDF text through Docling with OCR disabled. It reads TXT directly.
   - Text is normalized by `legal_chunker.normalize_for_lightrag` (embeds breadcrumbs, enforces `"\n\n"` around `Điều` markers).
   - Normalized text is sent to `RAGEngine.ainsert`.
   - `RAGEngine` stores chunks in PostgreSQL vector store (embeddings) and updates the knowledge graph.
2. **Query**
   - UI sends a `ChatRequest` (`message`, optional `history`, `stream`, `comparison_mode`).
   - `routes.chat` builds a system prompt enforcing strict answer style.
   - If `comparison_mode` is **false**, a single `rag.aquery` with `QueryParam(mode="hybrid")` is executed.
   - If **true**, two parallel queries (`naive` and `hybrid`) are launched, their streamed chunks merged into Server‑Sent Events.
   - LightRAG performs retrieval:
     - **Vector search** using the selected embedding backend.
     - **Graph search** leveraging entity types defined in `settings.ENTITY_TYPES`.
   - Retrieved passages are fed to the LLM wrapper (`deepseek_llm_func`).
   - LLM returns either a full string or an async generator for streaming.
   - API streams JSON‑encoded SSE events back to the frontend, which renders them live.
3. **Document Inventory**
   - `/documents` endpoint reads `doc_status` storage to list indexed files and their processing status.
   - Frontend polls this endpoint every 10 seconds to keep the UI in sync.

## Pre‑Processing Pipeline
- **File type detection** → TXT files are read directly. PDFs are converted through Docling with OCR disabled. Scanned/image-only PDFs fail clearly instead of entering an OCR fallback.
- **Text normalization** – `legal_chunker` adds breadcrumbs and ensures `\n\n` separators align with `Điều` boundaries.
- **Chunking** – LightRAG splits the normalized text by `"\n\n"`.
- **Embedding** – either OpenRouter (`QwenEmbeddingFunc`) or local Sentence‑Transformer (`VietLegalHarrierEmbeddingFunc`).
- **Metadata** – each chunk stores source file path, content summary, and timestamps.

## Model / LLM Details
- **LLM**: `gemini-3-flash-preview` accessed through OpenRouter (`settings.LLM_MODEL`).
- **Embedding backend**: configurable via `EMBEDDING_BACKEND` (default `sentence_transformers`).
- **Embedding model**: `mainguyen9/vietlegal-harrier-0.6b` (1024‑dim) when using local backend; otherwise `openai/text-embedding-3-small` (1536‑dim).
- **Prompt engineering** – a strict system prompt enforces answer‑only output, no pre‑ambles.

## Training / Evaluation
- The repository currently **does not contain training code** – LightRAG relies on pre‑trained embeddings and LLM.
- No explicit evaluation scripts are present; inference performance can be measured via the `/chat` endpoint and manual comparison mode.

## Checkpoint & Persistence
- LightRAG stores **vector embeddings** and **graph data** directly in PostgreSQL tables; there are no separate checkpoint files.
- The working directory (`settings.LIGHTRAG_WORKING_DIR`, default `./backend/data`) contains temporary index files and caches.
- Re‑indexing is required when changing the embedding dimension or model.

## Error Handling & Logging
- API routes raise `HTTPException` with status 500 on unexpected errors.
- Streaming errors are wrapped in SSE `type: error` messages.
- LLM wrapper prints debug messages to stdout; a proper logger is a known technical‑debt item.

## Extensibility Points
- **Embedding backend** – add new backends by extending `_build_embedding_func` in `RAGEngine`.
- **LLM wrapper** – replace `deepseek_llm_func` with other providers; ensure `extra_headers` are updated.
- **Prompt templates** – modify `system_prompt` in `routes.chat` for different instruction styles.
- **Entity taxonomy** – extend `settings.ENTITY_TYPES` to capture additional legal concepts.

---
*Future AI agents can reference this architecture file to understand component responsibilities, data flow, and where to plug in new models or storage backends.*
