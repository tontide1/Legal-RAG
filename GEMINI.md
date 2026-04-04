# GEMINI.md for Legal-RAG Project

## Project Overview

The Legal-RAG project is a Vietnamese legal question-answering system. It leverages a combination of technologies including:
- **Retrieval-Augmented Generation (RAG):** To provide relevant legal information.
- **Neo4j:** As a graph database to store and query legal documents and their relationships.
- **Named Entity Recognition (NER):** Specifically a BiLSTM model to identify legal spans like "Điều <number>".
- **Hybrid Retrieval:** Combines BM25 (keyword-based) and SBERT (semantic) retrieval with graph re-ranking for improved accuracy.
- **Gemini API:** Used for generating answers based on retrieved information, with `gemini-2.5-flash-lite` as the default model.

The architecture focuses on creating a knowledge graph from legal documents, enabling sophisticated querying and contextually relevant answers.

## Building and Running

### Prerequisites
- Python 3.11+
- Conda environment manager
- Docker

### Setup

1.  **Create and activate Conda environment:**
    ```bash
    conda create -n RAG python=3.11 -y
    conda activate RAG
    ```

2.  **Install Python dependencies:**
    ```bash
    python -m pip install --upgrade pip
    python -m pip install 
      torch numpy python-dotenv neo4j rank-bm25 sentence-transformers 
      langchain langchain-core langchain-google-genai huggingface-hub 
      torch-geometric
    ```

3.  **Configure environment variables:**
    Create a `.env` file in the project root with the following content, replacing placeholders with your actual credentials:
    ```env
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_neo4j_password
    GOOGLE_API_KEY=your_google_api_key
    GEMINI_MODEL=gemini-2.5-flash-lite
    HUGGINGFACEHUB_API_TOKEN=optional
    ```

4.  **Run Neo4j:**
    ```bash
    docker compose up -d
    ```
    Neo4j UI is accessible at `http://localhost:7474` (default user: `neo4j`, password from `.env`).

### Running the Project

1.  **Activate Conda environment:**
    ```bash
    conda activate RAG
    ```

2.  **Save data to Neo4j:**
    This script loads legal documents and structures them into the Neo4j graph.
    ```bash
    python src/save_database/save_data.py
    ```

3.  **Create embeddings:**
    Generates content and graph embeddings for efficient retrieval.
    ```bash
    python src/embedding/create_db.py
    ```

4.  **Run the QA system:**
    This starts the main CLI interface for asking legal questions.
    ```bash
    python src/main.py
    ```

### Testing

1.  **Activate Conda environment:**
    ```bash
    conda activate RAG
    ```

2.  **Run pipeline utility tests:**
    ```bash
    python -m unittest tests.test_pipeline_utils
    ```

3.  **Compile Python files:**
    ```bash
    python -m py_compile src/main.py src/save_database/save_data.py src/embedding/create_db.py
    ```

## Development Conventions

### Project Structure
- `src/`: Contains the core application logic.
    - `save_database/`: Scripts for loading data into Neo4j.
    - `embedding/`: Scripts for generating embeddings.
    - `retrive/`: Implements retrieval strategies.
    - `NER/`: Contains Named Entity Recognition models and processing.
    - `main.py`: The primary entry point for the CLI application.
- `tests/`: Unit tests for various components.
- `docs/`: Documentation files, including planning and reports.
- `dataset/`: Raw legal documents.
- `scripts/`: Utility scripts, such as skill validation.

### Technology Stack & Practices
- **Language:** Python 3.11+
- **Environment Management:** Conda
- **Database:** Neo4j (with specific labels like `LegalRAG` and node IDs in the format `{Label}::{Name}`)
- **LLM:** Gemini API (default: `gemini-2.5-flash-lite`)
- **NER:** BiLSTM model for specific legal entities (e.g., "Điều <number>")
- **Retrieval:** Hybrid approach (BM25, SBERT, Graph Reranking)
- **RAG Implementation:** Focus on graph-based RAG, inspired by NAGphormer.
- **Agent Skills:** Managed via `.codex/skills` and `.opencode/skills` directories, with validation scripts (`scripts/validate_skills.py`, `tests.test_skill_validation`).
- **Testing:** Uses Python's `unittest` framework.

### Notes on Implementation
- The NER model is currently specialized for identifying "Điều" clauses and may not perform general NER.
- API rate limits (e.g., Gemini's 429 error) are noted as potential issues.
- The current MVP is an approximation of the NAGphormer legal Graph RAG.
