---
name: legal-graph-rag
description: Project-specific operating guide for this Vietnamese Legal Graph RAG repository. Use for any task that touches legal ingestion, Neo4j graph state, hybrid retrieval, graph rerank, Gemini answers, or repo invariants.
risk: medium
source: personal
date_added: "2026-03-30"
---

# Legal Graph RAG

This is the project-specific skill for the repository.

## Use this skill when

- The task touches `src/main.py`, retrieval, embeddings, or Neo4j writes
- The task changes prompt assembly or final answer behavior
- You need to reason about repo invariants before making a change
- A generic RAG suggestion may conflict with how this repo actually works

## Non-negotiable repo invariants

- Main flow: `NER -> hybrid retrieval -> graph rerank -> LLM answer generation`
- Main UX and prompts should remain in Vietnamese
- Neo4j application data is scoped with label `LegalRAG`
- `node_id` is the stable unique identifier for graph write/read paths
- Do not key graph operations only by `ten`
- Normalize missing `Value` to `""`, not `None`
- Query embeddings and stored node embeddings must remain in the same embedding space
- Gemini model is resolved from `GEMINI_MODEL`
- Current repo default is `gemini-2.5-flash-lite`
- Destructive Neo4j cleanup must stay limited to app-owned nodes

## Operational checkpoints

- If source legal data changed:
  - `python src/save_database/save_data.py`
- If graph content or embeddings changed:
  - `python src/embedding/create_db.py`
- If validating the interactive path:
  - `python src/main.py`
- For lightweight regression:
  - `python -m unittest tests.test_pipeline_utils`

## Design defaults

- Prefer improving retrieval before changing Gemini prompts.
- Prefer graph-safe updates over broad cleanup or rewrite queries.
- Keep Vietnamese legal grounding stronger than stylistic fluency.
- If introducing a new model, index, or cache, define the revalidation path explicitly.

## Route to other skills

- `vietnamese-legal-nlp` for legal language structure and NER limits
- `rag-implementation` for end-to-end pipeline work
- `rag-engineer` for retrieval quality
- `gemini-api-dev` for Gemini integration

## Anti-patterns

- Generic RAG advice that ignores `node_id`
- Reindexing only part of the corpus after changing embeddings
- Treating duplicate `ten` values as unique entities
- Using destructive Cypher cleanup outside app-owned scope
