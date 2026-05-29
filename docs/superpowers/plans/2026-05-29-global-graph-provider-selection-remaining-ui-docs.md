# Global Graph Provider Selection Remaining UI and Docs Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining frontend, documentation, and end-to-end verification work for the global graph-provider feature on top of the backend that is already implemented in this branch.

**Architecture:** Keep backend/provider selection as already implemented, add a separate sidebar settings section for `graph_build_provider` with explicit `Save`, keep `FileUpload` focused on upload-only behavior, and document the runtime/configuration changes in repo docs. The frontend will treat the backend as the single source of truth and only persist changes through `PUT /api/settings/graph-provider`.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind CSS, FastAPI backend APIs already present, pytest.

---

## Scope Note

This follow-up plan assumes the current worktree already contains:

- PostgreSQL-backed graph-provider settings persistence
- provider-aware ingest selection in `RAGEngine`
- graph-provider settings API endpoints

This plan covers only the remaining UI, docs, and final verification work.

## File Structure

- Create: `frontend/src/components/GraphProviderSettings.tsx`
  - Own the load/save UX for the global graph-build provider, including unsaved state, save button, loading states, and save-time validation feedback.

- Modify: `frontend/src/App.tsx`
  - Add a dedicated sidebar settings section for graph-build configuration and pass the committed provider label down to the upload area.

- Modify: `frontend/src/components/FileUpload.tsx`
  - Keep upload-only behavior, but display the currently committed provider as passive context for future uploads.

- Modify: `.env.example`
  - Document the new graph-provider bootstrap/default and `NINE_ROUTER_*` variables.

- Modify: `README.md`
  - Document the global graph-provider setting, runtime behavior, and setup expectations.

- Create: `ARCHITECTURE.md`
  - Provide a concise architecture overview including app settings persistence and provider-aware ingest selection.

- Create: `PROJECT_STRUCTURE.md`
  - Document the key backend/frontend files involved in provider settings and upload/indexing flow.

---

### Task 1: Add a Separate Sidebar Settings Section for Graph Provider

**Files:**
- Create: `frontend/src/components/GraphProviderSettings.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Capture the current frontend baseline**

Run:

```bash
npm run build --prefix frontend
```

Expected: PASS with the current production build.

- [ ] **Step 2: Create the dedicated settings component**

Create `frontend/src/components/GraphProviderSettings.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle2, Loader2, Save } from 'lucide-react'
import client from '../api/client'

interface GraphProviderSettingsProps {
  onProviderCommitted?: (provider: string, label: string) => void
}

interface GraphProviderOption {
  value: string
  label: string
}

type SaveStatus = 'idle' | 'loading' | 'saving' | 'success' | 'error'

export default function GraphProviderSettings({ onProviderCommitted }: GraphProviderSettingsProps) {
  const [options, setOptions] = useState<GraphProviderOption[]>([])
  const [savedProvider, setSavedProvider] = useState('')
  const [selectedProvider, setSelectedProvider] = useState('')
  const [status, setStatus] = useState<SaveStatus>('loading')
  const [message, setMessage] = useState('')

  useEffect(() => {
    const loadProviderSettings = async () => {
      try {
        const [providerResponse, optionsResponse] = await Promise.all([
          client.get('/settings/graph-provider'),
          client.get('/settings/graph-provider/options'),
        ])

        const provider = providerResponse.data.provider as string
        const providerOptions = optionsResponse.data.options as GraphProviderOption[]
        const selectedOption = providerOptions.find((option) => option.value === provider)

        setOptions(providerOptions)
        setSavedProvider(provider)
        setSelectedProvider(provider)
        setStatus('idle')
        setMessage('')

        if (selectedOption && onProviderCommitted) {
          onProviderCommitted(selectedOption.value, selectedOption.label)
        }
      } catch (error: any) {
        setStatus('error')
        setMessage(
          error.response?.data?.detail ||
          error.message ||
          'Failed to load graph provider settings.'
        )
      }
    }

    void loadProviderSettings()
  }, [onProviderCommitted])

  const currentOption = options.find((option) => option.value === selectedProvider)
  const hasUnsavedChanges = selectedProvider !== '' && selectedProvider !== savedProvider

  const handleSave = async () => {
    if (!hasUnsavedChanges) return

    setStatus('saving')
    setMessage('')

    try {
      const response = await client.put('/settings/graph-provider', {
        provider: selectedProvider,
      })

      const provider = response.data.provider as string
      const committedOption = options.find((option) => option.value === provider)

      setSavedProvider(provider)
      setSelectedProvider(provider)
      setStatus('success')
      setMessage(response.data.message)

      if (committedOption && onProviderCommitted) {
        onProviderCommitted(committedOption.value, committedOption.label)
      }
    } catch (error: any) {
      setSelectedProvider(savedProvider)
      setStatus('error')
      setMessage(
        error.response?.data?.detail ||
        error.message ||
        'Failed to update graph provider.'
      )
    }
  }

  return (
    <div className="space-y-3 rounded-xl border bg-muted/30 p-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold text-foreground">Graph Build Provider</p>
          <p className="text-[10px] text-muted-foreground">
            Applies globally to all future uploads.
          </p>
        </div>
        {status === 'saving' && (
          <span className="flex items-center gap-1 text-[10px] font-medium text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            {selectedProvider === '9router' ? 'Checking 9router...' : 'Saving...'}
          </span>
        )}
      </div>

      <select
        value={selectedProvider}
        disabled={status === 'loading' || status === 'saving'}
        onChange={(event) => {
          setSelectedProvider(event.target.value)
          if (status === 'success') {
            setStatus('idle')
            setMessage('')
          }
        }}
        className="w-full rounded-lg border bg-background px-3 py-2 text-xs font-medium text-foreground"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>

      <div className="flex items-center justify-between gap-3">
        <p className="text-[10px] text-muted-foreground">
          {hasUnsavedChanges
            ? `Unsaved change: ${currentOption?.label ?? selectedProvider}`
            : `Current provider: ${currentOption?.label ?? 'Loading...'}`
          }
        </p>
        <button
          onClick={() => void handleSave()}
          disabled={!hasUnsavedChanges || status === 'loading' || status === 'saving'}
          className="inline-flex items-center gap-2 rounded-lg bg-primary px-3 py-2 text-[11px] font-bold text-primary-foreground transition-opacity disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Save className="h-3 w-3" />
          Save
        </button>
      </div>

      {status === 'success' && message && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50 p-2 text-[10px] font-bold text-emerald-600">
          <CheckCircle2 className="h-3 w-3" />
          <span>{message}</span>
        </div>
      )}

      {status === 'error' && message && (
        <div className="flex items-center gap-2 rounded-lg border border-red-100 bg-red-50 p-2 text-[10px] font-bold text-red-600">
          <AlertCircle className="h-3 w-3" />
          <span>{message}</span>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Wire the new settings section into the sidebar**

Update `frontend/src/App.tsx`:

```tsx
import { useState, useEffect } from 'react'
import ChatInterface from './components/ChatInterface'
import FileUpload from './components/FileUpload'
import GraphProviderSettings from './components/GraphProviderSettings'
import { Scale, Database, Shield, Share2, FileText, ExternalLink, Columns } from 'lucide-react'
import client from './api/client'

function App() {
  const [dbStatus, setDbStatus] = useState<'connected' | 'disconnected'>('disconnected')
  const [documents, setDocuments] = useState<any[]>([])
  const [comparisonMode, setComparisonMode] = useState(false)
  const [graphProviderLabel, setGraphProviderLabel] = useState('Ollama')

  const checkBackendHealth = async () => {
    try {
      await client.get('/health')
      setDbStatus('connected')
    } catch (error) {
      setDbStatus('disconnected')
    }
  }

  const fetchDocuments = async () => {
    try {
      const response = await client.get('/documents')
      setDocuments(response.data)
      setDbStatus('connected')
    } catch (error) {
      setDbStatus('disconnected')
      setDocuments([])
      console.error('Failed to fetch documents:', error)
    }
  }

  useEffect(() => {
    checkBackendHealth()
    fetchDocuments()

    const interval = setInterval(() => {
      fetchDocuments()
    }, 10000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden">
      <aside className="w-80 border-r bg-card p-6 flex flex-col gap-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Scale className="w-6 h-6 text-primary" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-foreground">Law Assistant</h1>
        </div>

        <div className="space-y-6 flex-1 overflow-y-auto pr-2 scrollbar-thin">
          <section>
            <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">RAG Settings</h2>
            <button
              onClick={() => setComparisonMode(!comparisonMode)}
              className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all ${
                comparisonMode
                  ? 'bg-primary/10 border-primary text-primary shadow-[0_0_12px_rgba(var(--primary),0.1)]'
                  : 'bg-muted/50 border-border text-muted-foreground hover:bg-muted'
              }`}
            >
              <div className="flex items-center gap-2">
                <Columns className="w-4 h-4" />
                <span className="text-sm font-medium">Comparison Mode</span>
              </div>
              <div className={`w-8 h-4 rounded-full p-1 transition-colors ${comparisonMode ? 'bg-primary' : 'bg-muted-foreground/30'}`}>
                <div className={`w-2 h-2 rounded-full bg-white transition-transform ${comparisonMode ? 'translate-x-4' : 'translate-x-0'}`} />
              </div>
            </button>
            <p className="text-[10px] text-muted-foreground mt-2 px-1">
              {comparisonMode ? 'Comparing Naive vs Hybrid RAG responses.' : 'Standard Hybrid RAG retrieval active.'}
            </p>
          </section>

          <section>
            <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Graph Build Settings</h2>
            <GraphProviderSettings
              onProviderCommitted={(_, label) => {
                setGraphProviderLabel(label)
              }}
            />
          </section>

          <section>
            <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">System Status</h2>
            <div className="flex items-center gap-2 text-sm font-medium">
              <div className={`w-3 h-3 rounded-full ${dbStatus === 'connected' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-red-500'}`} />
              <span className="text-foreground">DB: {dbStatus === 'connected' ? 'Connected' : 'Disconnected'}</span>
            </div>
          </section>

          <section>
            <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Knowledge Base</h2>
            <FileUpload onSuccess={fetchDocuments} providerLabel={graphProviderLabel} />
          </section>
        </div>
      </aside>

      <main className="flex-1 flex flex-col relative bg-muted/30">
        <ChatInterface comparisonMode={comparisonMode} />
      </main>
    </div>
  )
}

export default App
```

- [ ] **Step 4: Rebuild the frontend**

Run:

```bash
npm run build --prefix frontend
```

Expected: PASS.

- [ ] **Step 5: Commit the sidebar settings section**

```bash
git add frontend/src/components/GraphProviderSettings.tsx frontend/src/App.tsx
git commit -m "feat: add sidebar graph provider settings"
```

---

### Task 2: Keep File Upload Focused and Provider-Aware

**Files:**
- Modify: `frontend/src/components/FileUpload.tsx`

- [ ] **Step 1: Update the upload component props and passive provider hint**

Replace `frontend/src/components/FileUpload.tsx` with:

```tsx
import { useState } from 'react'
import { Upload, File, X, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import client from '../api/client'

interface FileUploadProps {
  onSuccess?: () => void
  providerLabel?: string
}

export default function FileUpload({ onSuccess, providerLabel = 'Ollama' }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState('')

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setStatus('idle')
      setMessage('')
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setStatus('uploading')
    setMessage('')
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await client.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      if (response.data?.status !== 'success') {
        throw new Error(response.data?.message || 'Upload failed')
      }

      setStatus('success')
      setMessage(response.data.message)
      setFile(null)
      if (onSuccess) onSuccess()
    } catch (error: any) {
      setStatus('error')
      setMessage(
        error.response?.data?.detail ||
        error.message ||
        'Cannot reach backend API. Make sure the backend is running on port 8000.'
      )
    }
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border bg-muted/20 px-3 py-2">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Future uploads
        </p>
        <p className="text-xs font-medium text-foreground">
          Graph build provider: {providerLabel}
        </p>
      </div>

      {!file ? (
        <label className="flex h-32 w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors group hover:bg-muted/50">
          <div className="flex flex-col items-center justify-center pb-6 pt-5">
            <Upload className="mb-2 h-6 w-6 text-muted-foreground transition-colors group-hover:text-primary" />
            <p className="text-xs font-medium text-muted-foreground">Click to upload legal PDFs or TXTs</p>
          </div>
          <input type="file" className="hidden" accept=".pdf,.txt" onChange={handleFileChange} />
        </label>
      ) : (
        <div className="flex items-center justify-between rounded-xl border bg-card p-3 animate-in zoom-in-95">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="rounded-lg bg-primary/10 p-2">
              <File className="h-4 w-4 text-primary" />
            </div>
            <span className="truncate text-xs font-medium">{file.name}</span>
          </div>
          <button onClick={() => setFile(null)} className="rounded-full p-1 hover:bg-muted">
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>
      )}

      {file && status === 'idle' && (
        <button
          onClick={handleUpload}
          className="w-full rounded-lg bg-primary py-2 text-xs font-bold text-primary-foreground transition-all hover:opacity-90 shadow-lg shadow-primary/20"
        >
          Begin Indexing
        </button>
      )}

      {status === 'uploading' && (
        <div className="flex items-center justify-center gap-2 py-2 text-xs font-medium text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>Building graph with {providerLabel}...</span>
        </div>
      )}

      {status === 'success' && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-100 bg-emerald-50 p-2 text-[10px] font-bold text-emerald-600">
          <CheckCircle2 className="h-3 w-3" />
          <span>{message}</span>
        </div>
      )}

      {status === 'error' && (
        <div className="flex items-center gap-2 rounded-lg border border-red-100 bg-red-50 p-2 text-[10px] font-bold text-red-600">
          <AlertCircle className="h-3 w-3" />
          <span>{message}</span>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Rebuild the frontend again**

Run:

```bash
npm run build --prefix frontend
```

Expected: PASS.

- [ ] **Step 3: Commit the upload-card refinement**

```bash
git add frontend/src/components/FileUpload.tsx
git commit -m "feat: show active graph provider in upload card"
```

---

### Task 3: Document the New Provider Workflow

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Create: `ARCHITECTURE.md`
- Create: `PROJECT_STRUCTURE.md`

- [ ] **Step 1: Add graph-provider and 9router variables to `.env.example`**

Update `.env.example` immediately below the existing Ollama block:

```dotenv
GRAPH_BUILD_PROVIDER_DEFAULT=ollama
NINE_ROUTER_BASE_URL=http://host.docker.internal:20128/v1
NINE_ROUTER_API_KEY=your_9router_api_key_here
NINE_ROUTER_INDEX_MODEL=cc/claude-sonnet-4-20250514
NINE_ROUTER_TIMEOUT_SECONDS=60
NINE_ROUTER_MAX_RETRIES=2
NINE_ROUTER_RETRY_DELAY_SECONDS=3
```

- [ ] **Step 2: Update `README.md` for the new global provider behavior**

Add or update the following sections in `README.md`:

```md
- **Configurable Graph Builder**: Choose `Ollama` or `9router Local` from the sidebar for future knowledge-graph indexing runs.
```

```md
- **LLM/Embeddings**: Gemini Developer API for chat generation, Ollama or 9router local for LightRAG indexing, Docling for no-OCR PDF text extraction, plus local Vietnamese legal embeddings with `huyydangg/DEk21_hcmute_embedding`
```

```md
## Graph Build Provider

The sidebar now includes a dedicated `Graph Build Settings` section:

- `Ollama` uses the local Ollama indexing model configured by `OLLAMA_*`
- `9router Local` uses the OpenAI-compatible local proxy configured by `NINE_ROUTER_*`

The selection is global, stored in PostgreSQL, and only applies to future uploads.
Changing to `9router Local` is validated when you click `Save`.
```

- [ ] **Step 3: Create `ARCHITECTURE.md` with the provider-setting architecture**

Create `ARCHITECTURE.md`:

```md
# Architecture

## Core Services

| Component | Responsibility | Key Files |
| --- | --- | --- |
| Backend API | Serves chat, upload, settings, and document inventory endpoints. | `backend/main.py`, `backend/api/routes.py` |
| App Settings Store | Persists global application settings such as `graph_build_provider` in PostgreSQL and validates 9router before saving it. | `backend/core/app_settings.py` |
| RAG Engine | Maintains one query engine plus provider-specific ingest engines selected at runtime. | `backend/core/rag_engine.py` |
| LLM Services | Provides Gemini chat, Ollama indexing, and OpenAI-compatible 9router indexing wrappers. | `backend/core/llm_services.py` |
| Frontend Sidebar | Displays comparison mode, graph-build settings, upload controls, and indexed-document status. | `frontend/src/App.tsx`, `frontend/src/components/GraphProviderSettings.tsx`, `frontend/src/components/FileUpload.tsx` |

## Graph Provider Flow

1. The frontend loads the current graph provider and available options from `/api/settings/graph-provider` and `/api/settings/graph-provider/options`.
2. The user changes the provider locally in the sidebar and clicks `Save`.
3. The backend validates `9router` only when saving that provider, then persists the committed value in PostgreSQL.
4. On upload, the backend reads the committed provider from the settings service and resolves the matching provider-specific ingest engine.
5. The query/chat flow remains Gemini-based and is not affected by the graph-build provider.
```

- [ ] **Step 4: Create `PROJECT_STRUCTURE.md` for the relevant provider files**

Create `PROJECT_STRUCTURE.md`:

````md
# Project Structure

## Backend

```text
backend/
├── api/
│   ├── routes.py          # Chat, upload, and graph-provider settings endpoints
│   └── schemas.py         # Request/response models including graph-provider payloads
├── core/
│   ├── app_settings.py    # PostgreSQL-backed global settings + provider options
│   ├── llm_services.py    # Gemini chat, Ollama ingest, 9router ingest
│   ├── rag_engine.py      # Query engine + provider-aware ingest cache
│   └── document_processor.py
└── tests/
    ├── test_graph_provider_settings.py
    ├── test_rag_engine.py
    └── test_upload_route.py
```

## Frontend

```text
frontend/src/
├── App.tsx                            # Sidebar layout and page wiring
├── api/client.ts                      # Axios API client
└── components/
    ├── GraphProviderSettings.tsx      # Separate sidebar settings section
    ├── FileUpload.tsx                 # Upload-only UI with passive provider hint
    └── ChatInterface.tsx
```
````

- [ ] **Step 5: Commit the docs**

```bash
git add .env.example README.md ARCHITECTURE.md PROJECT_STRUCTURE.md
git commit -m "docs: describe graph provider workflow"
```

---

### Task 4: Final Verification

**Files:**
- No source-file changes required unless verification finds a real defect.

- [ ] **Step 1: Run the full backend test suite**

Run:

```bash
PYTHONPATH=. pytest backend/tests -v
```

Expected: PASS across graph-provider settings, ingest runtime, upload route, and existing backend tests.

- [ ] **Step 2: Run the production frontend build**

Run:

```bash
npm run build --prefix frontend
```

Expected: PASS with emitted Vite output in `frontend/dist`.

- [ ] **Step 3: Manually verify the sidebar workflow**

Run:

```bash
docker compose up --build frontend backend
```

Expected:

- the sidebar shows a dedicated `Graph Build Settings` section separate from `Knowledge Base`
- changing the provider updates only local UI state until `Save` is clicked
- clicking `Save` with `9router` shows `Checking 9router...`
- failed 9router validation restores the previous saved value and shows the backend error
- successful save changes the provider used by the next upload

- [ ] **Step 4: Commit the final verification checkpoint**

```bash
git commit --allow-empty -m "chore: verify graph provider ui flow"
```

---

## Self-Review Coverage Map

- Separate sidebar settings section with explicit save: Task 1
- Upload card remains upload-only with passive provider context: Task 2
- `validate on Save` UX: Task 1 and Task 4
- README and environment variable documentation: Task 3
- Architecture and structure docs consistent with the implemented backend: Task 3
- Full backend/frontend verification: Task 4
