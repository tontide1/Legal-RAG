---
name: vector-database-engineer
description: Design vector storage and retrieval indexing for RAG systems. Use for Neo4j-stored embeddings, candidate retrieval strategy, metadata filtering, and tradeoffs around introducing a dedicated vector database.
risk: medium
source: personal
date_added: "2026-03-28"
---

# Vector Database Engineer

Use this skill for storage, indexing, and vector retrieval design.

## This repo first

- The project already uses Neo4j and stores graph-linked data there.
- Default assumption: keep retrieval close to the existing Neo4j-centered design unless the user explicitly wants a separate vector database.
- Any proposal to add Pinecone, Qdrant, Weaviate, or pgvector must justify migration cost, graph integration impact, and operational complexity.

## Use this skill when

- Changing how embeddings are stored or queried
- Tuning candidate retrieval latency or recall
- Adding metadata filters or hybrid search strategy
- Evaluating whether Neo4j is enough or a dedicated vector DB is needed

## Decision rules

1. Keep `node_id` as the stable link between vectors and graph entities.
2. Preserve metadata needed for legal filtering and reranking.
3. Optimize the candidate pool before touching the final rerank stage.
4. If storage changes, define how existing embeddings are migrated or rebuilt.

## For this project, check first

- Can the current Neo4j-backed setup satisfy recall and latency needs?
- Is the real issue chunking or embeddings rather than vector storage?
- Does a separate vector store complicate graph rerank and synchronization?

## Anti-patterns

- Adding a second vector store without a sync strategy
- Breaking graph-to-vector identity by dropping `node_id`
- Benchmarking final answers without measuring retrieval quality
- Changing storage without a reindex plan

## Related skills

- `legal-graph-rag`
- `rag-engineer`
- `embedding-strategies`

## Quick checklist

- Where are vectors stored now?
- What metadata is required for filtering and reranking?
- Does the proposal improve recall, latency, or maintainability enough to justify migration?
