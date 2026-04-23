---
name: gemini-api-dev
description: Build or debug Gemini API integrations for this repo. Use for model selection, SDK usage, generation settings, structured output, context caching, and migration decisions while preserving the project's Gemini defaults.
risk: medium
source: personal
date_added: "2026-03-28"
---

# Gemini API Development

Use this skill when working on Gemini integration, model selection, prompt wiring, caching, or generation bugs.

## Repo defaults

- This repository resolves the model from `GEMINI_MODEL`.
- The current project default is `gemini-2.5-flash-lite`.
- Do not replace the repo default just because newer preview models exist.
- Change model only when the user asks, benchmarks justify it, or official docs require migration.

## Source of truth

- Gemini model availability changes frequently. Verify on the official models page before making claims.
- Use the official docs index when implementing details:
  - `https://ai.google.dev/gemini-api/docs/models`
  - `https://ai.google.dev/gemini-api/docs/llms.txt`
- Treat hardcoded "latest model" claims as stale by default.

## Use this skill when

- Selecting a Gemini model for this project
- Debugging Gemini request or response behavior
- Migrating between Gemini SDKs or API versions
- Adding structured output, function calling, or context caching
- Tuning generation settings for Vietnamese legal QA

## Do not use this skill when

- The task is model-agnostic LLM architecture
- The system does not use Gemini
- You only need prompt-writing advice without Gemini integration details

## Preferred implementation rules

1. Preserve `GEMINI_MODEL` configurability in `src/main.py` and shared helpers.
2. Fail fast on missing `GOOGLE_API_KEY`.
3. Keep prompts and user-facing outputs in Vietnamese unless the task is explicitly technical.
4. Keep generation deterministic enough for legal QA:
   - prefer low temperature unless the user asks otherwise
   - prefer structured intermediate outputs when parsing citations or entities
5. If switching model families, document any required revalidation for latency, cost, and answer quality.

## SDK guidance

- Prefer current official Gemini SDKs and official docs over copied snippets.
- When editing existing repo code, preserve the working SDK unless migration is part of the task.
- If a migration is needed, verify request shapes and supported features against the official docs first.

## Project-specific sharp edges

- Do not "upgrade" away from `gemini-2.5-flash-lite` without checking repo constraints and user intent.
- Do not mix Vietnamese legal prompts with overly creative settings.
- Do not claim a model is deprecated unless the official models page says so.
- If retrieval or graph quality is poor, do not blame the model first. Check chunking, embeddings, candidate pool, and reranking.

## Related skills

- `legal-graph-rag` for repo architecture and invariants
- `rag-implementation` for end-to-end Legal Graph RAG workflow
- `prompt-caching` for Gemini caching decisions

## Quick checklist

- Which Gemini model is configured now?
- Is the bug in retrieval, prompt assembly, or Gemini output?
- Are official docs needed because model or SDK behavior may have changed?
- Does the change preserve Vietnamese legal answer quality?
