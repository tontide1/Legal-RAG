# AGENTS.md

Guide for agentic coding assistants working in this repository.

## 1) Project Overview
- Repository type: Vietnamese legal Graph RAG pipeline.
- End-to-end flow: `NER -> hybrid retrieval -> graph rerank -> LLM answer generation`.
- Product language: user-facing prompts and answers should remain in Vietnamese.
- Runtime style: script-first Python project (no Poetry/Makefile task runner).

## 2) Key Repository Paths
- `src/main.py`: interactive legal QA CLI entrypoint.
- `src/legal_qa.py`: orchestrates NER, retrieval, answer generation, and structured outputs.
- `src/pipeline_utils.py`: shared helpers (`node_id`, text payload, Gemini model selection).
- `src/save_database/save_data.py`: loads legal entities + relationships into Neo4j.
- `src/embedding/create_db.py`: builds and stores content + graph embeddings.
- `src/retrive/multi_retr.py`: BM25 + SBERT + graph rerank hybrid retrieval.
- `src/NER/ner.py`: BiLSTM NER training/inference.
- `evaluation/run_eval.py`: deterministic evaluation runner.
- `scripts/validate_skills.py`: validates `.codex/skills` and `.opencode/skills` parity.
- `tests/test_pipeline_utils.py`, `tests/test_legal_qa.py`, `tests/test_evaluation_metrics.py`, `tests/test_skill_validation.py`: core regression tests.

## 3) Environment and Dependencies
- Python: 3.11+ recommended.
- Preferred environment: `conda activate RAG`.
- Install dependencies: `python3 -m pip install -r requirements.txt`.
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
- `EVAL_ENABLE_LLM_JUDGE` and `EVAL_JUDGE_MODEL` (evaluation controls)

## 4) Build / Lint / Test Commands

### Setup and runtime
- Start Neo4j: `docker compose up -d`
- Ingest graph data: `python3 src/save_database/save_data.py`
- Rebuild embeddings: `python3 src/embedding/create_db.py`
- Run QA CLI: `python3 src/main.py`

Recommended rerun order after data/schema flow changes:
1. `python3 src/save_database/save_data.py`
2. `python3 src/embedding/create_db.py`
3. `python3 src/main.py`

### Test commands (unittest)
- Run all tests: `python3 -m unittest`
- Run one test module:
  - `python3 -m unittest tests.test_pipeline_utils`
  - `python3 -m unittest tests.test_legal_qa`
- Run one test class:
  - `python3 -m unittest tests.test_pipeline_utils.PipelineUtilsTest`
  - `python3 -m unittest tests.test_legal_qa.LegalQAPipelineTest`
- Run a single test method (important for fast debug loops):
  - `python3 -m unittest tests.test_pipeline_utils.PipelineUtilsTest.test_make_node_id_uses_label_and_name`
  - `python3 -m unittest tests.test_legal_qa.LegalQAPipelineTest.test_run_legal_qa_returns_abstain_when_retrieval_is_empty`
  - `python3 -m unittest tests.test_evaluation_metrics.EvaluationMetricsTest.test_ndcg_at_k_rewards_better_ranking`
  - `python3 -m unittest tests.test_skill_validation.SkillValidationTest.test_skill_validator_passes_for_codex_and_opencode_skills`

### Evaluation and validation
- Run deterministic evaluation (no LLM judge): `python3 evaluation/run_eval.py --disable-llm-judge`
- Validate skill trees + parity: `python3 scripts/validate_skills.py --repo-root .`
- Run skill tests: `python3 -m unittest tests.test_skill_validation`

### Lightweight syntax checks
- Quick syntax check:
  - `python3 -m py_compile src/main.py src/legal_qa.py src/save_database/save_data.py src/embedding/create_db.py src/retrive/multi_retr.py`

### Lint / formatter status
- No dedicated linter/formatter config (`ruff`, `black`, `isort`, `mypy`) is committed.
- Keep changes PEP 8-oriented and consistent with existing code style.
- Do not introduce a new formatting tool unless explicitly requested.

## 5) Code Style Guidelines

### Imports
- Order imports as: standard library -> third-party -> local modules.
- Group imports cleanly with one blank line between groups.
- Prefer absolute imports inside `src/` where practical.
- Remove unused imports during edits.

### Formatting
- Use 4-space indentation.
- Keep functions focused and testable; extract helpers for complex logic.
- Keep line lengths reasonable and readable (PEP 8 baseline).
- Add comments only for non-obvious logic or domain constraints.

### Typing
- Add type hints for new/modified public functions.
- Prefer concrete types (`list[dict]`, `dict[str, float]`) over broad `Any`.
- Keep function signatures stable unless the task requires an API change.

### Naming conventions
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Preserve dataset field names in Vietnamese when required by existing payload schema.

### Error handling
- Fail fast when required credentials or configuration are missing.
- Do not silently swallow database or external API failures.
- Raise explicit exceptions with actionable messages.
- If exceptions are transformed for UX, preserve technical details in diagnostics.

### Script boundaries
- Keep import-heavy or training-heavy execution behind `main()`.
- Use `if __name__ == "__main__":` for script entrypoints.

## 6) Data and DB Safety Rules
- Scope application graph operations to nodes labeled `LegalRAG`.
- Use parameterized Cypher values whenever possible.
- Be conservative with destructive operations; avoid broad cleanup beyond app-owned scope.
- Never commit secrets (`.env`, API keys, passwords, tokens).

## 7) Architecture Invariants (Do Not Break)
- `node_id` is the stable unique identifier for graph read/write operations.
- Do not key graph operations only by `ten` (duplicate names may exist).
- Normalize missing `Value` to `""` (empty string), not `None`.
- Retrieval stack remains hybrid:
  - BM25 lexical retrieval
  - SBERT semantic retrieval (`keepitreal/vietnamese-sbert`)
  - Graph rerank on a candidate pool larger than final `top_k`
- Query embeddings and stored node embeddings must stay in the same embedding space.
- `src/main.py` and `src/legal_qa.py` resolve Gemini model via `GEMINI_MODEL`.
- Current default Gemini model is `gemini-2.5-flash-lite`.

## 8) Current Model/Behavior Constraints
- NER label set currently: `O`, `B-ARTICLE`, `I-ARTICLE`.
- NER is strongest on references like `Điều <số>` and not yet a general legal-entity recognizer.
- Abstention behavior in QA should remain deterministic when retrieval is empty.

## 9) Agent Workflow Expectations
- Before edits, inspect related modules and preserve behavior unless change is required.
- For retrieval/graph changes, verify impact on node identity, embeddings, and reranking.
- For generation changes, prioritize legal grounding quality over stylistic fluency.
- After edits, run targeted tests first, then broader suites as needed.
- If data flow changes, document required rerun order clearly.

## 10) Skills and Agent Metadata
- Skill trees exist in both:
  - `.codex/skills/<name>/SKILL.md`
  - `.opencode/skills/<name>/SKILL.md`
- `.codex/skills` is source-of-truth; `.opencode/skills` must mirror 1:1.
- Validate parity with: `python3 scripts/validate_skills.py --repo-root .`.

## 11) Documentation Policy
- Keep `AGENTS.md` focused on durable repository constraints and operating guidance.
- Put session-specific progress/blockers/next steps in `docs/session_handoff.md`.
- Keep MVP/report planning in `docs/mvp_plan.md` and related `docs/` plans.
