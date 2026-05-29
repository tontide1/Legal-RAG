# Hybrid Jina Rerank Design

## Goal

Add reranking to the `hybrid` retrieval path using Jina API while keeping `naive` unchanged.

This design must:

- enable rerank only for `hybrid`
- keep `naive` as the exact-passage baseline with no rerank dependency
- remove the current LightRAG warning by making rerank behavior explicit per query mode
- integrate through LightRAG's supported `rerank_model_func` interface instead of building a custom rerank layer

## Scope

In scope:

- add Jina rerank configuration to backend settings and `.env.example`
- create a Jina-backed rerank wrapper for LightRAG
- attach rerank capability to the query-time `LightRAG` instance only
- force `naive` queries to run with `enable_rerank=False`
- force `hybrid` queries to run with explicit rerank policy
- add or update tests for query-param behavior and LightRAG initialization

Out of scope:

- changing frontend behavior or labels
- adding rerank to `naive`
- adding rerank to ingest/indexing
- replacing LightRAG retrieval logic with a custom rerank pipeline
- supporting multiple rerank providers in this change

## Current Behavior

The current backend creates query parameters without setting `enable_rerank`.

As a result:

- LightRAG defaults `enable_rerank=True`
- the repository does not configure any `rerank_model_func`
- both `naive` and `hybrid` trigger the LightRAG warning about missing rerank configuration
- retrieval continues to work, but reranking is never actually applied

This is a configuration mismatch, not a retrieval crash.

## Target Behavior

After this change:

- `naive` always executes with `enable_rerank=False`
- `hybrid` executes with `enable_rerank=True` only when hybrid rerank is enabled and Jina credentials are present
- if hybrid rerank is disabled or Jina credentials are missing, `hybrid` falls back to `enable_rerank=False`
- the backend logs a clear app-level message when hybrid rerank is configured off or unavailable
- the current generic LightRAG warning no longer appears in normal app usage because the app stops asking LightRAG to rerank without a reranker

## Chosen Architecture

Use one query-side `LightRAG` instance with Jina rerank capability attached, and control actual rerank behavior per query through `QueryParam.enable_rerank`.

Why this is the preferred design:

- it matches LightRAG's intended extension point
- it avoids creating duplicate query instances
- it keeps `naive` and `hybrid` separation at the behavior layer where the difference actually matters
- it minimizes code churn in a repository that already centralizes query orchestration through `RAGEngine` and `run_hybrid_query()`

## Configuration Design

Add the following settings:

- `JINA_API_KEY`
- `JINA_RERANK_MODEL`
- `JINA_RERANK_BASE_URL`
- `HYBRID_ENABLE_RERANK`

Recommended defaults:

- `JINA_RERANK_MODEL=jina-reranker-v2-base-multilingual`
- `JINA_RERANK_BASE_URL=https://api.jina.ai/v1/rerank`
- `HYBRID_ENABLE_RERANK=True`

Design rules:

- `JINA_API_KEY` is optional at configuration-load time so the app can still boot in environments that do not want rerank
- rerank availability is determined at runtime from both `HYBRID_ENABLE_RERANK` and presence of `JINA_API_KEY`
- this change does not rely on LightRAG's `RERANK_BY_DEFAULT` environment variable for app behavior

## Backend Integration

### Rerank Wrapper

Add a small wrapper around LightRAG's built-in Jina rerank helper.

Responsibilities:

- call the LightRAG Jina rerank function with configured model, API key, and base URL
- keep the signature compatible with what LightRAG expects from `rerank_model_func`
- avoid provider-specific logic leaking into route or query code

Preferred location:

- `backend/core/llm_services.py`

This repository already keeps model-service adapters there, so rerank belongs with the other external model bindings.

### RAG Engine

Update query-instance initialization in `backend/core/rag_engine.py`:

- attach `rerank_model_func` to the query `LightRAG` instance
- do not attach `rerank_model_func` to the ingest `LightRAG` instance

This keeps rerank capability available only where retrieval happens.

### Naive Query Path

Update query-param construction in `backend/api/routes.py` so `naive` explicitly sets:

- `enable_rerank=False`

This applies to:

- non-streaming `naive` requests in comparison mode
- streaming `naive` requests and their fallback path

### Hybrid Query Path

Update `backend/core/hybrid_query.py` so `QueryParam` is built with:

- `enable_rerank=True` when hybrid rerank is available
- `enable_rerank=False` when hybrid rerank is unavailable

Availability rule:

- available only if `HYBRID_ENABLE_RERANK=True` and `JINA_API_KEY` is non-empty

This keeps `hybrid` behavior explicit and prevents LightRAG from receiving a rerank request it cannot fulfill.

## Failure Policy

Do not fail application startup just because Jina rerank is unavailable.

Instead:

- if `HYBRID_ENABLE_RERANK=False`, `hybrid` runs without rerank
- if `HYBRID_ENABLE_RERANK=True` but `JINA_API_KEY` is missing, `hybrid` runs without rerank and logs a clear warning from the application layer

Rationale:

- this repository is used interactively in local and Docker setups
- failing startup over an optional retrieval-quality feature is unnecessarily disruptive
- the user asked to add rerank to `hybrid`, not to make rerank mandatory for the whole service lifecycle

Runtime Jina API errors during an active query remain request failures. This change does not add a second fallback path that silently bypasses rerank after the request has already opted into it.

## Testing Strategy

### `backend/tests/test_rag_engine.py`

Add coverage to confirm:

- the query `LightRAG` instance receives a non-null `rerank_model_func`
- the ingest `LightRAG` instance does not receive a rerank function

### `backend/tests/test_chat_route.py`

Add or update coverage to confirm:

- `_build_query_param(..., mode="naive")` sets `enable_rerank=False`

### `backend/tests/test_hybrid_query.py`

Add coverage to confirm:

- `run_hybrid_query()` sets `enable_rerank=True` when hybrid rerank is available
- `run_hybrid_query()` sets `enable_rerank=False` when hybrid rerank is disabled
- `run_hybrid_query()` sets `enable_rerank=False` when the Jina API key is missing

The tests should verify the actual `QueryParam` object passed into LightRAG-facing calls rather than only checking helper return values.

## Verification Plan

Implementation verification should include:

- targeted pytest for `backend/tests/test_rag_engine.py`
- targeted pytest for `backend/tests/test_chat_route.py`
- targeted pytest for `backend/tests/test_hybrid_query.py`

A manual hybrid request with Jina credentials configured should also be used to confirm the previous LightRAG warning no longer appears on the hybrid path when the environment supports that check.

## Non-Goals and Constraints

- Do not refactor the overall `hybrid` anchor-first strategy in this change
- Do not change retrieval ranking heuristics outside the addition of LightRAG reranking
- Do not introduce a second rerank code path outside LightRAG
- Do not wire rerank controls into the frontend

## Summary

This change adds Jina rerank in the smallest maintainable way:

- one query-side rerank binding
- explicit `naive` opt-out
- explicit `hybrid` opt-in
- graceful fallback when credentials are unavailable
- tests that lock in mode-specific behavior
