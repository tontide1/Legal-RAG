# Global Graph Provider Selection Design

## Goal

Add a global application setting that lets the user choose which provider builds the LightRAG knowledge graph during document indexing:

- `ollama`
- `9router`

The setting must:

- be changeable directly from the UI
- be stored durably in PostgreSQL so it survives backend restarts
- apply to all future uploads globally rather than per file
- validate `9router` reachability before saving that provider
- keep the existing query/chat flow unchanged

## Scope

In scope:

- Add a persistent app settings store in PostgreSQL
- Add backend APIs to read and update the global graph-build provider
- Add a UI control to display and change the provider
- Add a new OpenAI-compatible indexing backend for `9router local`
- Update upload/indexing flow to use the currently saved provider
- Add tests for settings persistence, provider validation, and provider-based ingest routing

Out of scope:

- Per-upload provider overrides
- Automatic fallback from `9router` to `ollama` during indexing
- Query/chat provider switching
- Editing `.env` from the UI
- Multi-user or per-user settings
- Managing 9router model lists dynamically from the UI

## Current Behavior

The backend currently creates two LightRAG instances during startup:

- query instance using Gemini chat generation
- ingest instance hardcoded to `ollama_index_llm_func`

Relevant current wiring:

- `backend/core/llm_services.py` defines `ollama_index_llm_func`
- `backend/core/rag_engine.py` hardcodes the ingest instance to Ollama
- `backend/api/routes.py` upload route retrieves a single ingest engine instance and calls `rag.ainsert(...)`
- `frontend/src/components/FileUpload.tsx` has no provider selection UI and always uploads directly

Because the ingest provider is fixed at initialization time, the current system cannot support changing graph-build providers from the UI.

## Proposed Architecture

### 1. Persistent app settings table

Add a small application-level settings table in PostgreSQL:

```sql
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Seed or lazily initialize:

- `key = 'graph_build_provider'`
- default `value = 'ollama'`

This table is separate from LightRAG storage tables because the selected provider is an application setting, not document or graph data.

Because the repository does not currently expose a migration framework, table creation should be done with an idempotent bootstrap step during backend startup or on first settings-service access.

### 2. Backend settings service

Add a small settings layer responsible for:

- reading `graph_build_provider`
- validating allowed values
- creating the default record if it does not yet exist
- updating `updated_at` on writes

The allowed values are initially:

- `ollama`
- `9router`

This logic should live outside the route handlers so the route remains thin and the same service can be reused from upload/indexing code.

### 3. Provider-aware ingest engine selection

Refactor `RAGEngine` so ingest is selected by provider at runtime instead of being hardcoded once at startup.

Recommended shape:

- keep the existing query instance behavior unchanged
- replace the single `_ingest_instance` with a provider-keyed cache such as `_ingest_instances: dict[str, LightRAG]`
- add `get_ingest_instance(provider: str)`
- build a provider-specific ingest instance on first use, then reuse it

Provider behavior:

- `ollama` keeps the current indexing path
- `9router` adds a new indexing path using an OpenAI-compatible client pointed at the local 9router proxy

This preserves the current separation between query and ingest while avoiding repeated engine construction on every upload.

### 4. 9router local indexing backend

Add a new async LLM wrapper for LightRAG indexing, parallel to `ollama_index_llm_func`.

Recommended implementation shape:

```python
client = openai.AsyncOpenAI(
    api_key=settings.NINE_ROUTER_API_KEY,
    base_url=settings.NINE_ROUTER_BASE_URL,
)
```

Use the OpenAI-compatible Chat Completions interface against local 9router with:

- `model=settings.NINE_ROUTER_INDEX_MODEL`
- `temperature=0.0`
- `stream=False`
- a conservative output-token limit aligned with current indexing behavior

Reuse the same prompt-building rules already used for extraction:

- keep the strict output rules appended to the indexing system prompt
- preserve malformed-output detection
- preserve retry behavior with provider-specific timeout/retry settings

This design treats 9router as a transport-compatible ingest backend, not as a replacement for the overall LightRAG flow.

Documentation verified from the current 9router docs:

- local base URL can be `http://localhost:20128/v1`
- tools can connect using OpenAI SDK `base_url` plus API key
- requests use the standard `/v1/chat/completions` contract

## Configuration Changes

Extend `backend/config.py` with settings for the new provider and defaults:

- `GRAPH_BUILD_PROVIDER_DEFAULT=ollama`
- `NINE_ROUTER_BASE_URL=http://host.docker.internal:20128/v1`
- `NINE_ROUTER_API_KEY`
- `NINE_ROUTER_INDEX_MODEL`
- `NINE_ROUTER_TIMEOUT_SECONDS`
- `NINE_ROUTER_MAX_RETRIES`
- `NINE_ROUTER_RETRY_DELAY_SECONDS`

Notes:

- `host.docker.internal` is the correct default for the backend container reaching a host-local 9router proxy
- non-Docker local runs can override this to `http://localhost:20128/v1`
- the upload flow should not read provider choice from `.env` once the database setting exists; `.env` only supplies the default bootstrap value and connection details

## API Design

### GET `/api/settings/graph-provider`

Returns the currently saved global provider:

```json
{
  "provider": "ollama"
}
```

If the settings row does not exist yet, the backend should initialize it using `GRAPH_BUILD_PROVIDER_DEFAULT` and return that value.

### GET `/api/settings/graph-provider/options`

Returns provider options for the UI:

```json
{
  "options": [
    { "value": "ollama", "label": "Ollama" },
    { "value": "9router", "label": "9router Local" }
  ]
}
```

This endpoint is optional from a pure backend perspective, but recommended so the frontend does not hardcode available providers forever.

### PUT `/api/settings/graph-provider`

Request:

```json
{
  "provider": "9router"
}
```

Behavior:

- reject any unsupported provider value with `400`
- when switching to `ollama`, save immediately
- when switching to `9router`, run a backend validation request against the configured local proxy before saving
- if validation passes, persist the new value and return it
- if validation fails, do not change the stored setting

Recommended success response:

```json
{
  "provider": "9router",
  "status": "success",
  "message": "Graph build provider updated to 9router."
}
```

Recommended failure response:

```json
{
  "detail": "9router local proxy is not reachable at http://host.docker.internal:20128/v1"
}
```

## 9router Validation Rules

When `PUT /api/settings/graph-provider` receives `provider=9router`, the backend must validate the configured local proxy before writing the setting to PostgreSQL.

Validation should use a real OpenAI-compatible request rather than only a socket or HTTP ping, because the goal is to prove that:

- the base URL is reachable
- the API key is accepted
- the configured model name is usable
- the backend can complete the exact SDK path that indexing depends on

Recommended validation request:

- use the OpenAI-compatible client configured for 9router
- call `chat.completions.create(...)`
- send a minimal prompt such as `"Respond with OK"`
- use the configured `NINE_ROUTER_INDEX_MODEL`
- force `temperature=0`
- force a very low `max_tokens`
- use a short timeout

Validation failure cases should include:

- DNS or connection failure
- timeout
- unauthorized or invalid API key
- unknown or unavailable model
- malformed API response

The stored provider must remain unchanged on any validation failure.

## Upload and Indexing Flow

The upload route remains the entrypoint for indexing and continues to:

- store the uploaded file temporarily
- extract text
- normalize legal structure
- call `rag.ainsert(...)`

The change is in how the route selects the ingest engine:

1. Read the current `graph_build_provider` from the settings service
2. Ask `RAGEngine` for the ingest instance for that provider
3. Run `ainsert(...)` with that provider-backed LightRAG instance

The upload route must not accept a provider from the frontend request body or multipart form. This avoids mismatch between client-side UI state and the authoritative global backend state.

The success message should mention the provider used, for example:

```text
File indexed using graph provider: 9router
```

## UI Design

Add a compact global provider control near the existing upload component.

UI behavior:

- on load, fetch current provider
- optionally fetch available options
- show a selector labeled `Graph Build Provider`
- when the user changes the value, call `PUT /api/settings/graph-provider`
- while saving, disable the selector
- when switching to `9router`, show a transient status like `Checking 9router...`
- if save succeeds, keep the new selection and show success feedback
- if save fails, revert to the previous value and show the backend error

The upload button does not allow per-file provider selection. It only reflects the current global state, for example:

- `Graph build provider: Ollama`
- `Graph build provider: 9router Local`

No dedicated settings page is required for this feature. The upload area is the correct place because the setting only affects indexing/build-graph behavior.

## Runtime Behavior and Caching

Recommended runtime behavior:

- initialize the query LightRAG instance at startup exactly as today
- lazily initialize provider-specific ingest instances when first requested
- cache one ingest instance per provider for reuse

This avoids rebuilding LightRAG for every upload while still allowing the provider to change globally between uploads.

Changing the saved provider does not migrate or rebuild already indexed graph data. It only changes which provider handles future entity/triple extraction during indexing.

## Error Handling Rules

- If `ollama` is selected and Ollama is unavailable at upload time, indexing fails with a clear upload error.
- If `9router` is selected and the proxy later becomes unavailable after the setting was previously saved, indexing fails with a clear upload error.
- Do not silently fall back from one provider to the other during indexing.
- Do not auto-rewrite the saved provider when an upload fails.
- Return explicit provider-aware messages so operators can tell which backend failed.

No silent fallback is important because graph extraction quality may differ between providers, and unexpected fallback would make indexing provenance unclear.

## Files To Change

- `backend/config.py`
- `backend/core/llm_services.py`
- `backend/core/rag_engine.py`
- `backend/api/routes.py`
- new backend settings service/module for app settings persistence
- frontend upload/settings UI components, likely including `frontend/src/components/FileUpload.tsx`
- backend tests for settings APIs and provider-based ingest selection
- README and architecture docs describing the new graph-build provider setting

## Test Plan

Backend unit and API tests:

1. Settings service returns `GRAPH_BUILD_PROVIDER_DEFAULT` when the row does not yet exist.
2. Settings service persists `graph_build_provider` in PostgreSQL.
3. `PUT /settings/graph-provider` rejects unsupported values.
4. `PUT /settings/graph-provider` saves `ollama` without remote validation.
5. `PUT /settings/graph-provider` validates `9router` before saving.
6. `PUT /settings/graph-provider` does not update the stored value when 9router validation fails.
7. `RAGEngine.get_ingest_instance("ollama")` returns the Ollama-backed ingest engine.
8. `RAGEngine.get_ingest_instance("9router")` returns the OpenAI-compatible 9router-backed ingest engine.
9. Upload route reads provider from persisted settings rather than request input.
10. Upload success message includes the provider used.
11. 9router indexing wrapper retries and malformed-output handling behave consistently with the current Ollama extraction wrapper.

Frontend tests if present or practical:

- initial provider load renders correctly
- provider selector disables while saving
- failed switch to `9router` restores the previous value and shows the backend error

Verification commands after implementation:

```bash
pytest backend/tests
```

If frontend tests exist or are added:

```bash
npm test
```

## Risks

- Adding an app settings table introduces a small amount of direct SQL or persistence code outside LightRAG abstractions.
- `host.docker.internal` may behave differently across environments, so local non-Docker runs must be documented clearly.
- Different providers may produce different extraction quality or formatting strictness, especially for Vietnamese legal entities and relationships.
- A successful 9router validation check does not guarantee future availability during upload; runtime failures must still be surfaced clearly.

## Success Criteria

- The UI shows and updates a global graph-build provider.
- The selected provider is stored durably in PostgreSQL and survives restart.
- Switching to `9router` fails fast if the local proxy is unreachable or misconfigured.
- Upload/indexing uses the saved provider without per-file overrides.
- Query/chat behavior remains unchanged.
