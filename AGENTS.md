# AGENTS.md

Guide for agentic coding assistants working in this repository.

## 1) Project Overview
- Repository type: Vietnamese legal Graph RAG pipeline.
- End-to-end flow: `NER -> hybrid retrieval -> graph rerank -> LLM answer generation`.
- Product language: user-facing prompts and answers should remain in Vietnamese.
- Primary runtime: Python scripts (no centralized build tool like Poetry/Make at the moment).

## 2) Important Paths
- `src/main.py`: interactive legal QA CLI entrypoint.
- `src/pipeline_utils.py`: shared utilities (`node_id`, text payload, Gemini model selection).
- `src/save_database/save_data.py`: import legal entities + relationships into Neo4j.
- `src/embedding/create_db.py`: build text and graph embeddings, then persist.
- `src/retrive/multi_retr.py`: BM25 + SBERT + graph rerank retrieval.
- `src/NER/ner.py`: BiLSTM NER training/inference.
- `src/create_relation/create_node_rela.py`: experimental relation extraction.
- `scripts/validate_skills.py`: validates `.codex/skills` and `.opencode/skills` parity.
- `tests/test_pipeline_utils.py`: regression tests for shared pipeline utilities.
- `tests/test_skill_validation.py`: regression tests for skill structure/sync.

## 3) Environment and Dependencies
- Python: 3.11+ recommended.
- Preferred env: `conda activate RAG`.
- Install deps: `python3 -m pip install -r requirements.txt`.
- Local infra: Neo4j via `docker compose up -d`.
- Neo4j default URI: `bolt://localhost:7687` unless overridden.

Required environment variables:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `GOOGLE_API_KEY`

Optional environment variables:
- `GEMINI_MODEL` (default: `gemini-2.5-flash-lite`)
- `HUGGINGFACEHUB_API_TOKEN` (experimental scripts)

## 4) Build / Lint / Test Commands

### Core runtime commands
- Start Neo4j: `docker compose up -d`
- Ingest graph data: `python3 src/save_database/save_data.py`
- Rebuild embeddings: `python3 src/embedding/create_db.py`
- Run QA CLI: `python3 src/main.py`

### Test commands (unittest)
- Run all tests: `python3 -m unittest`
- Run one test module: `python3 -m unittest tests.test_pipeline_utils`
- Run one test class: `python3 -m unittest tests.test_pipeline_utils.PipelineUtilsTest`
- Run a single test method:
  - `python3 -m unittest tests.test_pipeline_utils.PipelineUtilsTest.test_make_node_id_uses_label_and_name`
  - `python3 -m unittest tests.test_skill_validation.SkillValidationTest.test_skill_validator_passes_for_codex_and_opencode_skills`

### Skill validation checks
- Validate skill trees + parity: `python3 scripts/validate_skills.py --repo-root .`
- Run skill tests: `python3 -m unittest tests.test_skill_validation`

### Lightweight syntax/health checks
- Quick syntax check:
  - `python3 -m py_compile src/main.py src/save_database/save_data.py src/embedding/create_db.py`

### Lint/formatter status
- No dedicated linter/formatter config (`ruff`, `black`, `isort`, `mypy`) is currently committed.
- Follow repo conventions in this file; do not introduce a new formatting tool unless explicitly requested.

## 5) Code Style and Conventions

### Imports
- Order imports as: standard library -> third-party -> local modules.
- Prefer absolute imports inside `src/` where practical.
- Keep import blocks clean; avoid unused imports.

### Formatting
- Use 4-space indentation.
- Keep code close to PEP 8.
- Avoid overly long functions; split logic into testable helpers.
- Add comments only for non-obvious logic or domain constraints.

### Typing
- Add type hints for new or modified public functions.
- Favor precise types over `Any` where feasible.
- Keep signatures stable unless the task requires API changes.

### Naming
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Maintain existing Vietnamese field names in dataset payloads when required by schema.

### Error handling
- Fail fast when required credentials are missing.
- Do not silently swallow external service/database errors.
- Prefer explicit exceptions with actionable messages.
- If converting exceptions for UX, preserve useful details for debugging.

### Data and DB safety
- Use parameterized Cypher query values whenever possible.
- Be conservative with destructive graph operations.
- Never run broad cleanup beyond app-owned scope.
- Never commit secrets (`.env`, API keys, passwords, tokens).

## 6) Architecture Invariants (Do Not Break)
- Neo4j app data is scoped with label `LegalRAG`.
- `node_id` is the stable unique identifier for graph read/write paths.
- Do not key graph operations only by `ten` (duplicate names can exist).
- Normalize missing `Value` to `""`, not `None`.
- Retrieval stack is hybrid:
  - BM25 lexical retrieval
  - SBERT semantic retrieval (`keepitreal/vietnamese-sbert`)
  - Graph rerank on a candidate pool larger than final `top_k`
- Query embeddings and stored node embeddings must remain in the same embedding space.
- `src/main.py` resolves Gemini model from `GEMINI_MODEL`.
- Current default Gemini model is `gemini-2.5-flash-lite`.

## 7) Current Model/Behavior Constraints
- NER labels currently: `O`, `B-ARTICLE`, `I-ARTICLE`.
- NER is strongest on references like `Điều <số>`; not a general legal-entity recognizer yet.
- Import-heavy or training-heavy scripts should run behind `main()` and `if __name__ == "__main__":`.

## 8) Agent Workflow Expectations
- Before edits, inspect relevant files and preserve existing behavior unless task requires change.
- For retrieval/graph changes, verify effects on node identity, embeddings, and reranking.
- For generation changes, keep legal grounding quality above stylistic fluency.
- After edits, run targeted tests first, then broader checks if needed.
- If data flow changed, document required rerun order (save data -> embeddings -> QA).

## 9) Skills and Agent-Specific Metadata
- Skill trees currently exist in both:
  - `.codex/skills/<name>/SKILL.md`
  - `.opencode/skills/<name>/SKILL.md`
- `.codex/skills` is treated as source-of-truth; `.opencode/skills` must mirror 1:1.
- Validate parity with: `python3 scripts/validate_skills.py --repo-root .`.

## 10) Cursor / Copilot Rules Check
- `.cursor/rules/`: not found in this repository at time of writing.
- `.cursorrules`: not found.
- `.github/copilot-instructions.md`: not found.
- If these files are added later, treat them as higher-priority agent instructions and update this document.

## 11) Documentation Policy
- Keep `AGENTS.md` focused on durable repository knowledge and operating constraints.
- Put session-specific progress, blockers, and next actions in `docs/session_handoff.md`.
- Keep MVP/report planning details in `docs/mvp_plan.md`.
