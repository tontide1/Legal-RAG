---
name: rag-implementation
description: Implement or refactor the repo's Legal Graph RAG workflow: legal data ingestion, Neo4j persistence, embeddings, hybrid retrieval, graph rerank, and Gemini answer generation.
risk: medium
source: personal
date_added: "2026-03-28"
---

# Legal Graph RAG Implementation

Use this skill for end-to-end changes to the repository's RAG pipeline.

## Repo workflow

`NER -> hybrid retrieval -> graph rerank -> Gemini answer generation`

## Use this skill when

- Changing ingestion or graph persistence
- Rebuilding embeddings or retrieval logic
- Improving graph reranking
- Adjusting prompt assembly for grounded answers
- Debugging poor answer quality in the full pipeline

## Project workflow

### Phase 1: Data and graph integrity

- Confirm the graph still uses label `LegalRAG`.
- Preserve `node_id` as the stable identifier for graph writes and reads.
- Never key graph operations only by `ten`.
- Normalize missing `Value` to `""`.

### Phase 2: Embeddings

- Keep query embeddings and stored node embeddings in the same embedding space.
- If the embedding model changes, plan a full re-embed and retrieval revalidation.
- Preserve the repo's current Vietnamese retrieval assumptions unless the task explicitly changes them.

### Phase 3: Retrieval

- Keep candidate generation broader than final `top_k`.
- Preserve the hybrid nature of retrieval: lexical + semantic + graph rerank.
- Evaluate failures separately for:
  - lexical recall
  - semantic recall
  - graph rerank ordering

### Phase 4: Answer generation

- Keep the final UX and prompts in Vietnamese.
- Ground answers in retrieved legal evidence.
- Prefer conservative, citation-aware outputs over fluent but weakly grounded text.

## Default toolchain for this repo

- `legal-graph-rag` for repo-specific invariants
- `vietnamese-legal-nlp` for legal text structure and NER limits
- `embedding-strategies` for embedding-space decisions
- `vector-database-engineer` when retrieval storage/indexing changes
- `gemini-api-dev` for generation-side changes
- `prompt-caching` if repeated retrieval or prompting is the bottleneck

## Common failure modes

- Embeddings were regenerated with a different model but old vectors remain in Neo4j.
- Retrieval `top_k` is small and graph rerank gets too few candidates.
- Duplicate entities appear because graph identity used `ten` instead of `node_id`.
- Generated answers look good but are not grounded in retrieved law text.
- A destructive cleanup query touched more than app-owned nodes.

## Acceptance checklist

- Graph writes remain scoped to app-owned nodes.
- Retrieval still combines BM25, SBERT, and graph rerank.
- Gemini configuration still comes from `GEMINI_MODEL`.
- Vietnamese legal QA behavior remains intact.
