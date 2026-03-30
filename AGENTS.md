# AGENTS.md

Guide for coding agents working in this repository.

## Project Identity
- This repository is a Vietnamese legal Graph RAG pipeline.
- Main flow: `NER -> hybrid retrieval -> graph rerank -> LLM answer generation`.
- Main UX and prompts should remain in Vietnamese.

## Stable Project Layout
- `src/main.py`: interactive legal QA entrypoint.
- `src/pipeline_utils.py`: shared helpers for `node_id`, text payload building, and Gemini model config.
- `src/save_database/save_data.py`: import legal entities and relationships into Neo4j.
- `src/embedding/create_db.py`: create text embeddings and graph embeddings, then persist them.
- `src/retrive/multi_retr.py`: BM25 + SBERT + graph reranking retrieval.
- `src/NER/ner.py`: BiLSTM NER training and inference.
- `src/create_relation/create_node_rela.py`: experimental relation extraction.
- `tests/test_pipeline_utils.py`: lightweight regression tests for shared helpers.
- `docker-compose.yml`: local Neo4j service for development.
- `dataset/`: legal documents and structured datasets.
- `docs/review_fix_plan.md`: technical fix summary and remaining risks.
- `docs/evaluation_metrics_plan.md`: evaluation metrics and implementation roadmap.
- `docs/mvp_plan.md`: MVP scope and delivery plan for the university project.
- `docs/session_handoff.md`: latest session handoff for the next working session.

## Runtime Contract
- Python 3.11+ recommended.
- Preferred environment: `conda activate RAG`.
- Neo4j is expected at `bolt://localhost:7687` unless overridden.

Expected environment variables:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `GOOGLE_API_KEY`
- `GEMINI_MODEL` optional, current default `gemini-2.5-flash-lite`
- `HUGGINGFACEHUB_API_TOKEN` optional for experimental scripts

## Durable Architecture Decisions
- Neo4j application data is scoped with label `LegalRAG`.
- `node_id` is the stable unique identifier for graph write/read paths.
- Do not key graph operations only by `ten`; duplicate names can exist across legal documents.
- Text payload building must normalize missing `Value` to `""`, not `None`.
- Retrieval uses:
  - BM25 lexical score
  - SBERT semantic score with `keepitreal/vietnamese-sbert`
  - graph rerank over a candidate pool larger than final `top_k`
- Query embeddings and stored node embeddings must stay in the same embedding space.
- `src/main.py` resolves the Gemini model from `GEMINI_MODEL`; default is `gemini-2.5-flash-lite`.
- Destructive Neo4j cleanup must stay limited to app-owned nodes, not the whole database.

## Current Behavioral Constraints
- Neo4j node data depends on properties such as `node_id`, `ten`, `Value`, text embeddings, and graph embeddings.
- NER labels are currently limited to `O`, `B-ARTICLE`, `I-ARTICLE`.
- Current NER behavior is strongest on article references like `Điều <số>` and is not a general legal-entity recognizer.
- Import-heavy or training-heavy scripts should run behind `main()` and `if __name__ == "__main__":`.

## Operating Workflow
- If using local infrastructure, start Neo4j with `docker compose up -d`.
- If source legal data changed, run `python src/save_database/save_data.py`.
- If graph content or embeddings changed, run `python src/embedding/create_db.py`.
- Run interactive QA with `python src/main.py`.
- For targeted regression checks, run `python -m unittest tests.test_pipeline_utils`.

## Code and Safety Rules
- Keep imports ordered: standard library, third-party, local modules.
- Prefer absolute imports where practical inside `src/`.
- Use 4 spaces and keep code close to PEP 8.
- Add type hints for new or modified public functions.
- Fail fast on missing required credentials.
- Do not silently swallow external-service errors.
- Never commit `.env`, API keys, passwords, or tokens.
- Use parameterized Cypher queries for values.
- Be explicit and conservative with destructive operations.

## Documentation Policy
- Keep `AGENTS.md` limited to durable repository knowledge and operating constraints.
- Put session-specific progress, open tasks, blockers, and next actions in `docs/session_handoff.md`.
- Keep MVP delivery scope and project-report planning in `docs/mvp_plan.md`.
