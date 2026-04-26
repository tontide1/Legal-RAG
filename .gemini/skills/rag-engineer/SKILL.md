---
description: Design and debug retrieval-augmented generation systems. Use for chunking,
  candidate generation, reranking, grounding quality, and retrieval evaluation, especially
  for this Legal Graph RAG repository. Use this skill when Chunking hurts recall or
  context quality, Retrieval finds the wrong legal articles, Graph reranking behaves
  unexpectedly, Generated answers are fluent but weakly grounded, You need separate
  retrieval and generation evaluation.
name: rag-engineer
---

# RAG Engineer

Use this skill when the task is about retrieval quality, grounding, or RAG architecture.

## Focus for this repo

- Vietnamese legal documents
- Hybrid retrieval
- Graph-aware reranking
- Evidence-grounded Gemini answers

## Use this skill when

- Chunking hurts recall or context quality
- Retrieval finds the wrong legal articles
- Graph reranking behaves unexpectedly
- Generated answers are fluent but weakly grounded
- You need separate retrieval and generation evaluation

## Engineering rules

1. Diagnose retrieval before tuning the LLM.
2. Keep lexical, semantic, and graph contributions observable.
3. Evaluate candidate recall before rerank precision.
4. Treat embedding-space mismatches as critical failures.
5. Prefer structure-aware chunking for legal texts.

## Chunking guidance

- Prefer article-, clause-, and point-aware chunk boundaries over naive fixed windows.
- Preserve legal references like `Điều`, `Khoản`, and `Điểm`.
- Add enough overlap to keep citation context, but not enough to flood retrieval with near-duplicates.

## Retrieval guidance

- Use BM25 for explicit legal terms and citation matching.
- Use semantic retrieval for paraphrased legal questions.
- Use graph rerank only after enough candidates are collected.
- Inspect failures with the candidate pool, not only the final answer.

## Anti-patterns

- Using the same `top_k` for candidate generation and final output
- Changing the LLM when recall is the real problem
- Mixing embeddings from different models
- Ignoring duplicate or ambiguous legal entity names

## Related skills

- `legal-graph-rag`
- `rag-implementation`
- `vietnamese-legal-nlp`
- `embedding-strategies`

## Quick checklist

- Is the failure in chunking, retrieval, rerank, or generation?
- Did query and stored embeddings come from the same model?
- Is graph rerank operating on enough candidates?
- Does the answer cite the right legal evidence?