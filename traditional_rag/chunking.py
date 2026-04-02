import os
from pathlib import Path
from typing import List
import chardet

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_and_chunk_documents(
        data_dir: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[Document]:

    """
    Loads all .txt files from a directory and splits them into chunks.

    Args:
        data_dir (str): The directory to load documents from.
        chunk_size (int): The size of each text chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        List[Document]: A list of document chunks.
    """
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        raw_text = f.read(1000)
                    encoding = chardet.detect(raw_text)['encoding']
                    loader = TextLoader(file_path, encoding=encoding)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Loaded and chunked {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    chunks = load_and_chunk_documents(data_dir=BASE_DIR / "dataset")