---
name: llm-app-patterns
description: High-level design patterns for LLM applications. Use for broad architecture choices, but prefer project-specific skills first for this Legal Graph RAG repository.
risk: medium
source: personal
date_added: "2026-03-28"
---

# LLM App Patterns

Use this skill for high-level architecture decisions across LLM apps.

## In this repo, prefer these skills first

- `legal-graph-rag` for project invariants
- `rag-implementation` for end-to-end pipeline changes
- `rag-engineer` for retrieval quality
- `gemini-api-dev` for Gemini integration
- `prompt-caching` for caching

## Use this skill when

- Comparing RAG vs agentic flows
- Choosing where retrieval, memory, and tools should live
- Designing observability or evaluation at the application level
- Refactoring a broad LLM workflow, not one narrow subsystem

## Core patterns

### Retrieval-first QA

- Retrieve evidence before generation.
- Keep retrieval quality independently measurable.
- Prefer grounded answers for legal and compliance domains.

### Graph-aware retrieval

- Use graph structure when relationships matter.
- Keep entity identity stable across ingestion, retrieval, and answer generation.
- Separate candidate generation from graph rerank.

### Conservative generation

- Use structured intermediate steps when legal citations matter.
- Keep prompts and outputs consistent with the target language and domain.
- Avoid creative defaults for factual legal QA.

## Anti-patterns

- Treating a weak retriever as a prompt problem
- Hiding retrieval failures behind fluent answers
- Mixing project-specific rules into a giant generic skill instead of a dedicated skill

## Quick checklist

- Is this an app-level architecture question or a subsystem fix?
- Should the answer be driven by retrieval, graph logic, or tool use?
- Which project-specific skill should handle the detailed implementation?
