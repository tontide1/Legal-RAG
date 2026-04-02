import os
import faiss
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from .chunking import load_and_chunk_documents


# Use the same embedding model as the main project for consistency
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
VECTOR_STORE_PATH = "traditional_rag/vector_store/faiss_index_traditional"


def create_and_save_embeddings(chunks: list):
    """
    Creates a FAISS vector store from documents in a directory and saves it.
    """
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store already exists at {VECTOR_STORE_PATH}. Skipping creation.")
        return

    # 1. check Chunk
    if not chunks:
        print("No documents were chunked. Aborting embedding creation.")
        return

    # 2. Embed
    print(f"Creating embeddings using '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Get the embedding dimension from the model
    sample_embedding = embeddings.embed_query("test")
    embedding_dim = len(sample_embedding)

    # 3. Store
    print("Creating FAISS vector store...")
    # Using a flat index for simplicity. For larger datasets, a more advanced index might be better.
    index = faiss.IndexFlatL2(embedding_dim)
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=chunks)

    # 4. Save
    print(f"Saving vector store to '{VECTOR_STORE_PATH}'...")
    vector_store.save_local(VECTOR_STORE_PATH)
    print("Vector store created and saved successfully.")

    print("Embedding dimension:", index.d)
    print("Stored vectors:", index.ntotal)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    chunks = load_and_chunk_documents(data_dir=(BASE_DIR.parent)/ "dataset")
    create_and_save_embeddings(chunks)