# AGENTS.md

Guide for coding agents working in this repository.
This codebase is a Vietnamese legal RAG pipeline with Neo4j + NER + hybrid retrieval + LLM answer generation.

## Project Layout
- `Code/main.py`: interactive legal question answering entrypoint.
- `Code/save_database/save_data.py`: import legal entities/relations JSON into Neo4j.
- `Code/embedding/create_db.py`: generate and save content embeddings and graph embeddings.
- `Code/retrive/multi_retr.py`: BM25 + SBERT + graph reranking retrieval.
- `Code/NER/ner.py`: BiLSTM NER training/inference for article/entity spans.
- `Code/create_relation/create_node_rela.py`: experimental relation extraction via HuggingFaceHub.
- `dataset/`: raw legal documents and structured JSON datasets.

## Runtime Requirements
- Python 3.11+ recommended.
- Neo4j database reachable from local environment.
- Env using conda (conda activate RAG)

Expected env vars:
- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD` (required)
- `GOOGLE_API_KEY` (required for `Code/main.py`)
- `HUGGINGFACEHUB_API_TOKEN` (optional; used in experimental script)

## Setup Commands
Run from repository root.

```bash
conda create -n RAG python=3.11 -y
conda activate RAG
python -m pip install --upgrade pip
python -m pip install \
  torch numpy python-dotenv neo4j rank-bm25 sentence-transformers \
  langchain langchain-core langchain-google-genai huggingface-hub \
  torch-geometric
```

Conda-first rule:
- Always activate environment before any run/lint/test command: `conda activate RAG`.

Notes:
- `torch`/`torch-geometric` may need CPU/GPU-specific install variants.
- No pinned dependency file currently exists; lock versions before production use.

## Build / Run Commands
This repository is script-driven (no formal build system).

```bash
# Load/refresh legal graph data in Neo4j
python Code/save_database/save_data.py

# Compute node embeddings and graph embeddings, then save to Neo4j
python Code/embedding/create_db.py

# Run interactive legal QA flow
python Code/main.py
```

## Lint / Format / Type Check
No lint/type config is committed yet. Use these defaults if needed.

```bash
python -m pip install ruff black mypy
ruff check Code
black --check Code
mypy Code
```

## Test Commands
Current status: no `tests/` directory is present yet.
Recommended test runner when tests are added:

```bash
python -m pip install pytest

# Run all tests
pytest

# Run one file
pytest tests/test_retrieval.py

# Run one test class
pytest tests/test_retrieval.py::TestHybridRetriever

# Run one single test
pytest tests/test_retrieval.py::TestHybridRetriever::test_rerank_scores

# Run tests matching keyword
pytest -k "ner and not slow"
```

Interim smoke checks (useful now):

```bash
# Import smoke test for NER module
python -c "from Code.NER import ner; print('ok')"

# Execute NER training/inference script
python Code/NER/ner.py
```

## Agent Workflow
1. Validate `.env` and Neo4j connectivity.
2. If dataset changed, run `save_data.py` first.
3. If graph/node content changed, run `create_db.py`.
4. Validate end-to-end behavior via `main.py`.
5. For NER modifications, run `Code/NER/ner.py` or at least an import/infer smoke test.

## Code Style Guidelines

### Imports
- Order imports: standard library, third-party, local modules.
- Prefer one import per line.
- Avoid wildcard imports.
- Prefer absolute imports where practical; keep local import style consistent inside `Code/`.

### Formatting
- Follow PEP 8.
- Use 4 spaces, no tabs.
- Preferred max line length: 100 (up to 120 when legal text forces it).
- Keep functions focused; split long procedural blocks where feasible.
- Preserve explicit `encoding="utf-8"` for Vietnamese text files.

### Types
- Add type hints for new or modified public functions.
- Add return annotations.
- Use `Optional[...]` or `| None` for nullable values.
- For structured dict payloads, prefer `TypedDict` when refactoring.

### Naming
- Modules/functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Keep domain naming explicit (`entity`, `relationship`, `embedding`, `query_text`).

### Error Handling
- Fail fast on missing required credentials (current pattern: `ValueError`).
- Handle external boundary failures (Neo4j, model API) with actionable messages.
- Do not swallow exceptions silently.
- Keep user-facing CLI errors concise and recovery-oriented.

### Data and Security
- Never commit `.env`, API keys, tokens, or passwords.
- Avoid printing sensitive values in logs.
- Treat model checkpoints and large binaries intentionally.

### Neo4j Safety
- Use parameterized Cypher values for properties.
- Sanitize dynamic labels/relationship names (existing scripts already strip backticks).
- Be cautious with destructive operations like `MATCH (n) DETACH DELETE n`.

### ML / Retrieval Practices
- Document scoring changes in retrieval/rerank logic.
- Keep training and inference responsibilities clearly separated.
- Cache heavy model loads if startup latency becomes problematic.

## Project Behaviors to Preserve
- Main UX and prompts are Vietnamese.
- Data flow depends on Neo4j graph + node properties (`ten`, `Value`, embeddings).
- Retrieval fuses BM25 lexical score and semantic score before graph rerank.
- NER labels currently use `O`, `B-ARTICLE`, `I-ARTICLE`.

## Cursor / Copilot Rules
Checked locations:
- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

Current status:
- No Cursor rule files found.
- No Copilot instructions file found.

If these files are added later, treat them as repository-level policy and update this guide.

## Skills Status (Short)
- Installed agent skills: none detected in this environment.
- If skills are added later, list their names and intended usage here.
