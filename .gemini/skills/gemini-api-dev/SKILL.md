---
name: gemini-api-dev
description: Build or debug Gemini API integrations. Use this skill when working on Gemini integration, model selection, SDK usage, generation settings, structured output, context caching, or tuning generation settings for Vietnamese legal QA.
---

# Gemini API Development

## Overview

Expert in Gemini API integration, model selection, and optimization. This skill focuses on building robust integrations with Google's Gemini models, emphasizing efficient SDK usage, structured outputs, and performance tuning while maintaining repo-specific configurations.

## Repo Defaults & Configuration

- **Model Resolution**: Models MUST be resolved from the `GEMINI_MODEL` environment variable.
- **Default Model**: The current project default is `gemini-2.5-flash-lite`.
- **API Keys**: Fail fast if `GOOGLE_API_KEY` is missing.
- **Model Selection**: Change models only when justified by benchmarks or official deprecation notices.

## Implementation Guidelines

- **Determinism**: Keep generation settings (e.g., low temperature) suitable for legal QA to ensure consistent answers.
- **Structured Output**: Prefer structured intermediate outputs when parsing citations, entities, or legal facts.
- **Language**: Maintain prompts and user-facing outputs in Vietnamese unless the task is explicitly technical.
- **Efficiency**: Utilize context caching and proper generation settings to optimize cost and latency.

## SDK & API Usage

- **Official Sources**: Always prefer current official Gemini SDKs and documentation (e.g., `https://ai.google.dev/gemini-api/docs`).
- **Migration**: When migrating SDKs or models, verify request shapes and supported features against official documentation first.
- **Revalidation**: Document revalidation paths for latency, cost, and quality when switching model families.

## Anti-patterns

- **Stale Claims**: Claiming a model is deprecated without verifying the official models page.
- **Hardcoded Models**: Using hardcoded model strings instead of `GEMINI_MODEL`.
- **Creative Settings**: Using high temperature or creative generation settings for formal legal tasks.
- **Misplaced Blame**: Blaming the model for poor results before checking retrieval, chunking, or embedding quality.

## Quick Checklist

- [ ] Which Gemini model is currently configured in the environment?
- [ ] Is the issue in retrieval, prompt assembly, or the model's output?
- [ ] Does the proposed change preserve Vietnamese legal answer quality?
- [ ] Have the official docs been consulted for recent API or SDK changes?
