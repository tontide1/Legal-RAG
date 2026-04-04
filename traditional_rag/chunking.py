from __future__ import annotations

from typing import List

import nltk
from nltk.tokenize import sent_tokenize

# Download punkt tokenizer if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextChunker:
    """Class for chunking text documents into smaller pieces."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str = "") -> List[dict]:
        """
        Split text into chunks with overlap.

        Args:
            text: The text to chunk
            doc_id: Optional document identifier

        Returns:
            List of chunk dictionaries with text, start/end positions, and doc_id
        """
        if not text.strip():
            return []

        chunks = []
        start = 0

        while start < len(text):
            # Find the end of this chunk
            end = min(start + self.chunk_size, len(text))

            # If we're not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence boundaries within the last 100 characters
                search_start = max(start, end - 100)
                sentences = sent_tokenize(text[search_start:end])

                if len(sentences) > 1:
                    # Use the end of the last complete sentence
                    last_sentence = sentences[-1]
                    sentence_end = search_start + text[search_start:end].rfind(last_sentence) + len(last_sentence)
                    if sentence_end > start:
                        end = sentence_end

            # Extract the chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start": start,
                    "end": end,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{len(chunks)}" if doc_id else f"chunk_{len(chunks)}"
                })

            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text):
                break

        return chunks

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries with 'text' and 'doc_id' keys

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            doc_id = doc.get("doc_id", "")

            chunks = self.chunk_text(text, doc_id)
            all_chunks.extend(chunks)

        return all_chunks


def chunk_text_simple(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple function to chunk text into smaller pieces.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    chunker = TextChunker(chunk_size, overlap)
    chunks = chunker.chunk_text(text)
    return [chunk["text"] for chunk in chunks]