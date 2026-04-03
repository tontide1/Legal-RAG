# AGENTS.md

Operational guide for agentic coding assistants in this repository.

## 1) Repo Snapshot
- Project: Vietnamese Legal Graph RAG for legal QA.
- Core pipeline invariant: `NER -> hybrid retrieval -> graph rerank -> Gemini answer`.
- Main orchestration: `src/legal_qa.py`.
- CLI entrypoint: `src/main.py`.
- Streamlit entrypoint: `streamlit_app.py`.
- UI runtime adapter: `src/ui_runtime.py`.
- User-facing language should remain Vietnamese unless task asks otherwise.

## 2) High-Value Paths
- `src/legal_qa.py`: callable QA pipeline and output contract.
- `src/retrive/multi_retr.py`: BM25 + SBERT + graph rerank retrieval.
- `src/NER/ner.py`: unified NER interface.
- `src/NER/phobert_ner.py`: PhoBERT inference and cache behavior.
- `src/save_database/save_data.py`: Neo4j ingestion.
- `src/embedding/create_db.py`: content + graph embedding build.
- `src/pipeline_utils.py`: shared helpers and Gemini model default.
- `streamlit_app.py`: one-page UI for MVP demo.
- `src/ui_runtime.py`: env status + UI wrapper to call pipeline.
- `tests/`: unit and smoke tests.

## 3) Environment and Setup
- Recommended env: `conda activate RAG`.
- Install dependencies: `python3 -m pip install -r requirements.txt`.
- Start local Neo4j: `docker compose up -d`.
- Required env vars:
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`
  - `GOOGLE_API_KEY`
- Optional env vars:
  - `GEMINI_MODEL` (default `gemini-2.5-flash-lite`)
  - `NER_BACKEND` (`phobert` default)
  - `PHOBERT_NER_CHECKPOINT`
  - `PHOBERT_NER_MAX_LENGTH`
  - `NER_DATASET_PATH`

## 4) Build / Run Commands

### Core pipeline (CLI)
- `docker compose up -d`
- `python3 src/save_database/save_data.py`
- `python3 src/embedding/create_db.py`
- `python3 src/main.py`

### Streamlit demo
- `streamlit run streamlit_app.py`

### Recommended data refresh order
1. `python3 src/save_database/save_data.py`
2. `python3 src/embedding/create_db.py`
3. `python3 src/main.py` or `streamlit run streamlit_app.py`

## 5) Test Commands (including single-test)

### Run all tests
- `python3 -m unittest`

### Run one module
- `python3 -m unittest tests.test_legal_qa`
- `python3 -m unittest tests.test_pipeline_utils`
- `python3 -m unittest tests.test_streamlit_smoke`
- `python3 -m unittest tests.test_ner_inference`

### Run one class
- `python3 -m unittest tests.test_legal_qa.LegalQAPipelineTest`
- `python3 -m unittest tests.test_streamlit_smoke.StreamlitSmokeTest`

### Run one test method (fastest loop)
- `python3 -m unittest tests.test_legal_qa.LegalQAPipelineTest.test_run_legal_qa_returns_abstain_when_retrieval_is_empty`
- `python3 -m unittest tests.test_pipeline_utils.PipelineUtilsTest.test_get_configured_gemini_model_uses_default_flash_lite`
- `python3 -m unittest tests.test_streamlit_smoke.StreamlitSmokeTest.test_check_env_status_returns_dict`

### Evaluation and helpers
- `python3 evaluation/run_eval.py --disable-llm-judge`
- `python3 scripts/validate_skills.py --repo-root .`

### Quick syntax sanity
- `python3 -m py_compile src/main.py src/legal_qa.py src/ui_runtime.py streamlit_app.py`

## 6) QA Output Contract (Do Not Break)
- `run_legal_qa(query)` returns a dict with keys:
  - `query`
  - `ner_entities`
  - `retrieved_nodes`
  - `context_text`
  - `answer_text`
  - `citations`
  - `scores`
  - `timings`
  - `errors`
- This contract is validated in `tests/test_legal_qa.py`.

## 7) Retrieval and Embedding Invariants
- Keep hybrid retrieval behavior intact:
  - BM25 lexical score
  - SBERT cosine score (`keepitreal/vietnamese-sbert`)
  - graph rerank on candidate pool
- Keep query embeddings and stored node embeddings in same embedding space.
- If embedding model or graph structure changes, rebuild embeddings before evaluation.

## 8) Graph and Data Safety
- Neo4j app data must stay scoped to label `LegalRAG`.
- `node_id` is the stable identifier; do not rely only on `ten`.
- Missing `Value` must normalize to `""`, not `None`.
- Prefer parameterized Cypher.
- Never commit secrets (`.env`, API keys, credentials).

## 9) Streamlit-Specific Notes
- `src/ui_runtime.py` should load `.env` before env checks.
- Env status fields for secrets should display `SET` or `MISSING`.
- Keep UI resilient with `try/except` around pipeline call.
- Avoid duplicating business logic in UI layer.
- Keep CLI (`src/main.py`) behavior unchanged.

## 10) Code Style Guidelines

### Imports
- Group order: standard library -> third-party -> local modules.
- Keep one blank line between groups.
- Remove unused imports.

### Formatting
- Follow PEP 8 style with 4-space indentation.
- Prefer small, focused functions.
- Keep diffs minimal and consistent with existing style.
- Add comments only when logic is non-obvious.

### Typing
- Add type hints for new and changed public APIs.
- Prefer concrete types (`list[str]`, `dict[str, str]`).
- Avoid broad `Any` unless unavoidable.

### Naming
- `snake_case` for variables and functions.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for constants.
- Preserve existing schema field names when required for compatibility.

### Error Handling
- Fail fast on missing required config.
- Raise actionable errors with clear context.
- Do not silently swallow infra/API exceptions.
- For user-facing flows, provide safe fallback plus diagnostics.

## 11) Lint / Formatter Status
- No enforced repo config for `ruff`, `black`, `isort`, or `mypy`.
- Do not introduce new tooling unless explicitly requested.
- Match existing style and keep changes pragmatic.

## 12) Agent Workflow Expectations
- Read related modules before cross-cutting edits.
- Preserve pipeline invariants unless task explicitly changes architecture.
- Run targeted tests first, then broader suites.
- When changing defaults or runtime behavior, update docs and tests together.
- Prefer updating `README.md` and planning docs when user-facing commands change.

## 13) Common Failure Modes
- `Neo.ClientError.Security.Unauthorized`: Neo4j credentials mismatch.
- `GOOGLE_API_KEY not found`: missing Gemini key in environment.
- `429 RESOURCE_EXHAUSTED`: Gemini quota/rate limit hit.
- If data changed but retrieval looks stale, rerun:
  - `python3 src/save_database/save_data.py`
  - `python3 src/embedding/create_db.py`

## 14) Documentation Policy
- Keep AGENTS.md durable and operational.
- Put temporary session notes in `docs/session_handoff/`.
- Keep planning/task status in `docs/planning/`.
- If commands, defaults, or runtime shape changes, update AGENTS.md promptly.
