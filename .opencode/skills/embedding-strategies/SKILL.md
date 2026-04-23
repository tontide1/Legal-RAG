---
name: embedding-strategies
description: Select and validate embedding approaches for RAG systems. Use for model choice, multilingual embeddings, re-embedding plans, and embedding-space integrity in this Legal Graph RAG project.
risk: medium
source: personal
date_added: "2026-03-28"
---

# Embedding Strategies

Use this skill for embedding-model decisions and embedding-quality debugging.

## Repo-specific rules

- Query embeddings and stored node embeddings must remain in the same embedding space.
- If the embedding model changes, plan a full re-embed and retrieval verification.
- Do not mix old and new vectors in Neo4j.

## Use this skill when

- Choosing an embedding model for Vietnamese legal retrieval
- Investigating semantic search failures
- Reducing embedding cost or latency
- Comparing multilingual and domain-specific options
- Planning a reindex or re-embed

## Default guidance for this repo

- Respect the existing retrieval stack and benchmark against it.
- For Vietnamese legal text, prefer models proven on Vietnamese or multilingual semantic retrieval.
- Do not switch to Gemini embeddings or another family unless the task includes a full migration plan.

## Selection criteria

- Language fit: Vietnamese legal queries and documents
- Retrieval quality: article-level and clause-level relevance
- Latency and cost
- Operational cost of re-embedding existing data
- Compatibility with current storage and retrieval code

## Anti-patterns

- Comparing models without using the same evaluation set
- Mixing vectors from multiple embedding models
- Changing dimensions or model family without reindexing
- Optimizing token cost while silently harming recall

## Related skills

- `legal-graph-rag`
- `rag-engineer`
- `vector-database-engineer`

## Quick checklist

- What model produced the stored vectors now?
- What model will produce query vectors after the change?
- Is there a full re-embed plan?
- How will retrieval quality be re-measured?
