# Traditional RAG

This folder contains the standalone traditional RAG pipeline used for local
chunking, embedding, retrieval, and Gemini-based response generation.

## Setup

Install the Python dependencies:

```bash
pip install -r traditional_rag/requirements.txt
```

Before running the pipeline, install the NLTK `punkt` tokenizer as a deployment
prerequisite:

```bash
python -m nltk.downloader punkt
```

The chunking code no longer downloads `punkt` at import time. If `punkt` is
missing, the pipeline will raise a clear runtime error telling you to install it
during setup.
