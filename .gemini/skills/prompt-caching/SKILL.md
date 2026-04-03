---
description: Design caching for LLM and RAG systems. Use for prompt-prefix caching,
  embedding caching, retrieval-result caching, response caching, and invalidation
  strategies that fit this Legal Graph RAG project. Use this skill when Reducing repeated
  embedding cost, Avoiding repeated retrieval on unchanged legal data, Reusing stable
  prompt prefixes or system instructions, Designing invalidation after dataset or
  graph updates, Tuning latency and cost for Gemini-backed QA.
name: prompt-caching
---

# Prompt Caching

Use this skill for cache design in LLM, RAG, and Graph RAG systems.

## Priority order for this repo

1. Embedding cache
2. Retrieval-result cache
3. Prompt-prefix or context cache
4. Final-answer cache only when grounding is stable

## Use this skill when

- Reducing repeated embedding cost
- Avoiding repeated retrieval on unchanged legal data
- Reusing stable prompt prefixes or system instructions
- Designing invalidation after dataset or graph updates
- Tuning latency and cost for Gemini-backed QA

## Do not use this skill when

- Data changes frequently and freshness matters more than latency
- Answers depend on per-user state or volatile graph writes
- The bottleneck is not repeated computation

## Cache layers

### 1. Embedding cache

- Cache by normalized text plus embedding model identifier.
- Invalidate when the embedding model changes.
- For this repo, query embeddings and stored node embeddings must stay in the same embedding space.

### 2. Retrieval-result cache

- Cache by normalized query, retrieval config, candidate pool size, and active dataset version.
- Include whether graph reranking is enabled.
- Invalidate when legal documents, graph structure, or node embeddings change.

### 3. Prompt-prefix or context cache

- Cache stable instruction prefixes and repeated legal context blocks.
- Best fit when the same system prompt or static legal framing is reused many times.
- Tie cache keys to the active Gemini model because cache behavior is model-specific.

### 4. Final-answer cache

- Use only for highly repeated, low-volatility questions.
- Key by normalized question, retrieved evidence set, and model.
- Avoid if answers depend on fresh graph state, user role, or changing documents.

## Invalidation rules for this project

Invalidate relevant caches when any of the following changes:

- source legal dataset files
- Neo4j content written by `save_data.py`
- node text embeddings or graph embeddings
- retrieval parameters, rerank logic, or `top_k`
- `GEMINI_MODEL`
- system prompt or answer template

## Anti-patterns

- Caching answers without including retrieved evidence in the key
- Reusing embeddings across different embedding models
- Caching retrieval before graph rerank while pretending the final result is cached
- Treating stale legal answers as acceptable
- Adding caching before measuring the actual hotspot

## Related skills

- `legal-graph-rag` for repo invariants
- `rag-implementation` for retrieval pipeline design
- `gemini-api-dev` for Gemini-specific caching behavior

## Quick checklist

- What exact computation is being repeated?
- What input fields define cache correctness?
- What event invalidates the cache?
- Is stale legal guidance acceptable? Usually no.