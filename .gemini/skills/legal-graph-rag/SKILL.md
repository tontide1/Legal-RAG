---
name: legal-graph-rag
description: Project-specific operating guide for the Vietnamese Legal Graph RAG repository. Use this skill when touching legal ingestion, Neo4j graph state, hybrid retrieval, graph reranking, Gemini answer generation, or maintaining repository invariants.
---

# Legal Graph RAG

## Overview

Core project guide for the Legal Graph RAG system. This skill ensures that all changes respect the repository's architectural invariants, naming conventions, and retrieval-generation workflows, particularly for Vietnamese legal content and Neo4j graph integration.

## Repo Invariants

- **Main Workflow**: NER -> Hybrid Retrieval -> Graph Rerank -> LLM Answer Generation.
- **Stable Identifiers**: Use `node_id` as the stable unique identifier for all graph paths. Do not use name strings (`ten`) as unique keys.
- **Data Scoping**: Neo4j application data is scoped with the `LegalRAG` label.
- **Vietnamese Language**: Prompts, UI text, and final answers MUST remain in Vietnamese.
- **Grounding**: Prioritize Vietnamese legal grounding over stylistic fluency.
- **Embedding Spaces**: Query embeddings and stored node embeddings MUST share the same embedding space.

## Implementation Standards

- **Value Handling**: Normalize missing `Value` to an empty string (`""`), NOT `None`.
- **Graph Updates**: Use graph-safe incremental updates over broad destructive cleanup queries.
- **Model Config**: Resolve the Gemini model from `GEMINI_MODEL`, defaulting to `gemini-2.5-flash-lite`.
- **Retrieval First**: Improve retrieval quality before adjusting Gemini prompts.

## Operational Checkpoints

- **Data Ingestion**: Use `python src/save_database/save_data.py` after legal source changes.
- **Index Rebuild**: Run `python src/embedding/create_db.py` after graph or embedding changes.
- **Integration Test**: Verify the main path with `python src/main.py`.
- **Unit Testing**: Run `python -m unittest tests.test_pipeline_utils` for regressions.

## Anti-patterns

- **Generic Advice**: Using generic RAG patterns that ignore `node_id` or Neo4j-specific labels.
- **Partial Indexing**: Reindexing only part of the corpus after changing embedding models.
- **Over-reaching Cleanup**: Using destructive Cypher cleanup outside the app-owned scope (`LegalRAG`).
- **Ignoring Duplicate Names**: Treating duplicate `ten` values as unique without checking `node_id`.

## Quick Checklist

- [ ] Does the change maintain the core pipeline (NER -> retrieval -> rerank -> answer)?
- [ ] Is the Neo4j query scoped correctly to the `LegalRAG` label?
- [ ] Are prompts and responses in the correct language (Vietnamese)?
- [ ] Does the change use `node_id` for graph node identification?
