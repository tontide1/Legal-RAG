---
name: systematic-debugging
description: Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes. Emphasizes root-cause investigation and evidence-first debugging.
risk: medium
source: personal
date_added: "2026-03-28"
---

# Systematic Debugging

## Core rule

No fixes before root-cause investigation.

## Use this skill when

- Tests fail
- Retrieval quality regresses
- Neo4j writes produce unexpected graph state
- Gemini output changes unexpectedly
- The pipeline works but the final answer is wrong

## Required workflow

1. Reproduce and capture the exact symptom.
2. Isolate the failing stage:
   - NER
   - ingestion
   - embeddings
   - hybrid retrieval
   - graph rerank
   - Gemini generation
3. Compare expected vs actual data flowing across that boundary.
4. Form one hypothesis at a time.
5. Add or update a failing test before fixing when practical.

## Project-specific guidance

- If answer quality drops, separate retrieval failure from generation failure.
- If graph results look wrong, inspect `node_id`, labels, and destructive cleanup logic first.
- If semantic retrieval degrades, verify embedding model consistency before changing prompts.
- If legal citations are wrong, inspect chunk boundaries and retrieval candidates before blaming Gemini.

## Supporting files in this skill

- `root-cause-tracing.md`
- `defense-in-depth.md`
- `condition-based-waiting.md`

## Related skills

- `debugging-strategies`
- `test-driven-development`
- `legal-graph-rag`

## Red flags

- "Quick fix first, investigate later"
- "Maybe the model changed"
- "Let's just increase top_k and hope"
- "Let's rewrite prompts before checking retrieval"
