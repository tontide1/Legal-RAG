---
description: Project-specific guidance for Vietnamese legal text processing. Use for
  legal citation parsing, chunking by article structure, NER limitations, normalization,
  and retrieval behavior on Vietnamese law documents. Use this skill when Improving
  legal citation extraction, Adjusting chunk boundaries for luật, nghị định, thông
  tư, or nghị quyết, Normalizing Vietnamese legal references before retrieval, Explaining
  why retrieval misses the right điều, khoản, or điểm.
name: vietnamese-legal-nlp
---

# Vietnamese Legal NLP

Use this skill when working on Vietnamese legal language, citation patterns, chunking, or NER behavior.

## Current repo reality

- NER labels are currently limited to `O`, `B-ARTICLE`, `I-ARTICLE`
- Current NER behavior is strongest on article references like `Điều <số>`
- This is not yet a general legal-entity recognizer

## Use this skill when

- Improving legal citation extraction
- Adjusting chunk boundaries for luật, nghị định, thông tư, or nghị quyết
- Normalizing Vietnamese legal references before retrieval
- Explaining why retrieval misses the right điều, khoản, or điểm

## Text-handling rules

1. Preserve legal markers such as `Điều`, `Khoản`, and `Điểm`.
2. Prefer chunking that follows legal document hierarchy over fixed-size windows.
3. Normalize superficial formatting differences without destroying citation meaning.
4. Treat citation extraction, retrieval, and answer generation as separate evaluation surfaces.

## Retrieval implications

- BM25 helps when users mention explicit legal references.
- Semantic retrieval helps with paraphrased legal questions.
- Graph rerank is only useful if chunks preserve the right legal units.

## Anti-patterns

- Splitting a clause across chunks without overlap
- Assuming the current NER model recognizes all legal entities
- Removing Vietnamese diacritics if the downstream behavior depends on exact legal text
- Evaluating legal QA only by answer fluency

## Related skills

- `legal-graph-rag`
- `rag-engineer`
- `embedding-strategies`