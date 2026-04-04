---
name: embedding-strategies
description: Select and validate embedding approaches for RAG systems. Use this skill when choosing an embedding model for Vietnamese legal retrieval, investigating semantic search failures, reducing embedding cost or latency, comparing multilingual and domain-specific options, or planning a reindex/re-embed.
---

# Embedding Strategies

## Overview

Expert in embedding-model decisions, quality debugging, and maintaining semantic integrity in RAG systems, particularly for Vietnamese legal content. This skill ensures retrieval quality by selecting appropriate models and managing embedding transitions.

## Repo-specific Rules

- **Space Consistency**: Query embeddings and stored node embeddings MUST remain in the same embedding space.
- **Migration Protocol**: If the embedding model changes, a full re-embed and retrieval verification MUST be planned.
- **Isolation**: Do not mix old and new vectors in the vector database (e.g., Neo4j).

## Implementation Guidelines

- **Vietnamese Legal Retrieval**: For Vietnamese legal text, prefer models proven on Vietnamese or multilingual semantic retrieval.
- **Selection Criteria**: Evaluate models based on language fit, article/clause-level relevance, latency, and re-embedding costs.
- **Evaluation**: Always compare models using the same evaluation set before making decisions.

## Anti-patterns

- **Blind Comparisons**: Comparing models without using the same evaluation set.
- **Inconsistent Spaces**: Mixing vectors from multiple embedding models in the same retrieval flow.
- **Silent Degradation**: Optimizing token cost while silently harming recall.
- **Broken Dimensions**: Changing dimensions or model family without reindexing.

## Quick Checklist

- [ ] What model produced the currently stored vectors?
- [ ] What model will produce query vectors after the change?
- [ ] Is there a comprehensive full re-embed plan?
- [ ] How will retrieval quality be measured post-change?
