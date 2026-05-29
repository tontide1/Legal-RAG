# Architecture

Traffic Law Assistant uses a FastAPI backend, a React frontend, and PostgreSQL for graph/vector storage.

## Global Graph Provider

- The sidebar exposes a dedicated `Graph Build Settings` section.
- The selected provider is stored in PostgreSQL as a global setting.
- `ollama` is the default graph-build provider.
- `9router` is validated before the backend saves the change.

## Runtime Flow

1. The user changes the provider in the sidebar.
2. The frontend saves the choice through the settings API.
3. The backend validates `9router` before persisting it.
4. Future uploads use the saved provider when building the graph.

Query and chat still use the dedicated query pipeline, but hybrid queries can now apply Jina reranking when configured. The graph-build provider setting only changes the indexing path used during upload.

## Main Files

- `backend/api/routes.py`: upload, document, and settings endpoints
- `backend/core/rag_engine.py`: provider-aware ingest selection
- `backend/core/llm_services.py`: chat, embedding, and indexing LLM wrappers
- `backend/config.py`: environment-driven defaults
- `frontend/src/App.tsx`: sidebar layout
- `frontend/src/components/GraphProviderSettings.tsx`: global graph provider UI
