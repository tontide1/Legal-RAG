# Project Structure

```text
.
├── backend/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── llm_services.py
│   │   └── rag_engine.py
│   ├── config.py
│   └── tests/
│       ├── test_chat_route.py
│       ├── test_document_processor.py
│       ├── test_graph_provider_settings.py
│       ├── test_rag_engine.py
│       └── test_upload_route.py
├── frontend/
│   └── src/
│       ├── App.tsx
│       └── components/
│           ├── FileUpload.tsx
│           └── GraphProviderSettings.tsx
├── ARCHITECTURE.md
├── PROJECT_STRUCTURE.md
├── README.md
└── .env.example
```

Relevant files for the graph provider setting:

- `frontend/src/components/GraphProviderSettings.tsx`: sidebar control for the global provider
- `frontend/src/components/FileUpload.tsx`: upload flow that uses the saved provider
- `backend/api/routes.py`: settings API and upload routing
- `backend/core/rag_engine.py`: provider-aware ingest selection
- `backend/config.py`: defaults and 9router configuration
