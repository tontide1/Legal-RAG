# Gemini LLM Fallback Implementation Design

## Goal

Replace the current OpenRouter-backed chat/insert LLM path with direct Gemini API usage using `gemini-2.0-flash-lite`, while keeping OpenRouter as an automatic fallback.

## Scope

In scope:

- Add a direct Gemini LLM wrapper that matches the current LightRAG callback shape.
- Route LightRAG's `llm_model_func` through Gemini first, then OpenRouter fallback on failure.
- Keep existing embedding behavior and document ingestion flow unchanged.
- Add config for Gemini API key, model name, and output-token limits.
- Add tests for Gemini primary path and OpenRouter fallback.

Out of scope:

- Changing embedding providers.
- Changing the upload OCR pipeline.
- Removing OpenRouter completely.
- UI changes.

## Current Behavior

The backend currently wires `backend.core.llm_services.deepseek_llm_func` into LightRAG as `llm_model_func`.

That wrapper uses OpenRouter through `openai.AsyncOpenAI`, so any LightRAG insert/query path that needs generation goes through OpenRouter.

This is the source of the current credit and max-token failures during document ingestion.

## Proposed Architecture

### 1. Gemini-first LLM wrapper

Create a Gemini-backed async wrapper in `backend/core/llm_services.py` that keeps the same external contract as the existing LightRAG callback:

- input: `prompt`, optional `system_prompt`, optional `history`, `**kwargs`
- output: generated text or async stream compatible with the current caller expectations

The wrapper will:

- use the direct Google Gemini SDK
- send the configured model name `gemini-2.0-flash-lite`
- apply a conservative `max_output_tokens` cap
- translate the current chat-style inputs into Gemini request format

### 2. Fallback wrapper

Add a higher-level wrapper that tries Gemini first and falls back to the existing OpenRouter implementation if Gemini fails.

Fallback should activate for transient provider errors such as:

- rate limits
- 5xx responses
- timeouts
- SDK/network failures

The fallback keeps the system usable when Gemini is unavailable, but it stays secondary to Gemini.

### 3. Config-driven provider settings

Extend `backend/config.py` with Gemini-specific settings:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `GEMINI_MAX_OUTPUT_TOKENS`

Keep existing OpenRouter settings for fallback.

### 4. LightRAG wiring

Update `backend/core/rag_engine.py` so LightRAG receives the new Gemini-first fallback wrapper instead of `deepseek_llm_func` directly.

No other RAG initialization behavior should change.

## Data Flow

1. User uploads or queries content.
2. LightRAG calls the configured `llm_model_func`.
3. The Gemini-first wrapper sends the request to Gemini directly.
4. If Gemini succeeds, the response is returned immediately.
5. If Gemini fails with a fallback-worthy error, the wrapper retries through OpenRouter.
6. LightRAG continues using the returned text as before.

## Error Handling Rules

- If Gemini API key is missing, fail fast during startup or first use with a clear configuration error.
- If Gemini returns a transient failure, fall back to OpenRouter.
- If both providers fail, raise the original/final error to LightRAG.
- Do not silently swallow generation errors.

## Testing Plan

1. Unit test Gemini wrapper request mapping.
2. Unit test Gemini wrapper streaming path if LightRAG consumes stream responses.
3. Unit test fallback to OpenRouter when Gemini raises a transient failure.
4. Unit test config wiring in `RAGEngine.initialize()`.
5. Regression test that LightRAG still receives a callable LLM function.

## Success Criteria

- Gemini API is the primary generation provider.
- OpenRouter is only used as fallback.
- Upload and query flows continue to work through LightRAG.
- Ingestion no longer depends on OpenRouter as the first-choice LLM provider.
