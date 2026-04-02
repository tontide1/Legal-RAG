from .chunking import load_and_chunk_documents
from .embedding import create_and_save_embeddings
from .llm_response import make_response

import os
from pathlib import Path


VECTOR_STORE_PATH = "traditional_rag/vector_store/faiss_index_traditional"

if __name__ == "__main__":
    if not os.path.exists(VECTOR_STORE_PATH):
        BASE_DIR = Path(__file__).resolve().parent
        chunks = load_and_chunk_documents(data_dir=(BASE_DIR.parent)/ "dataset")
        create_and_save_embeddings(chunks)

    print("\nType 'q' to quit.")
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'q':
            break

        make_response(question)
        print()