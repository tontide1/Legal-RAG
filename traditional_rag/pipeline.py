from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .chunking import TextChunker
from .embedding import TextEmbedder
from .llm_response import LLMResponseGenerator


class TraditionalRAGPipeline:
    """Complete RAG pipeline combining chunking, embedding, and LLM response generation."""

    def __init__(
        self,
        dataset_root: str | Path = "dataset",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "keepitreal/vietnamese-sbert",
        llm_model: str = "gemini-2.5-flash-lite",
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.

        Args:
            dataset_root: Root directory containing documents
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of embedding model
            llm_model: Name of LLM model
            top_k: Number of top chunks to retrieve
        """
        self.dataset_root = Path(dataset_root)
        self.top_k = top_k

        # Initialize components
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = TextEmbedder(model_name=embedding_model)
        self.llm_generator = LLMResponseGenerator(model_name=llm_model)

        # Load and process documents
        self.documents = self._load_documents()
        self.chunks = self._chunk_documents()
        self.chunk_texts = [chunk["text"] for chunk in self.chunks]
        self.chunk_embeddings = self.embedder.encode_texts(self.chunk_texts)

        print(f"Loaded {len(self.documents)} documents, created {len(self.chunks)} chunks")

    def _load_documents(self) -> List[dict]:
        """Load documents from the dataset directory."""
        documents = []
        if not self.dataset_root.exists():
            print(f"Warning: Dataset root {self.dataset_root} does not exist")
            return documents

        for path in sorted(self.dataset_root.rglob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    documents.append({
                        "doc_id": str(path.relative_to(self.dataset_root)),
                        "title": path.stem,
                        "source": str(path),
                        "text": text,
                    })
            except OSError as e:
                print(f"Error reading {path}: {e}")
                continue

        return documents

    def _chunk_documents(self) -> List[dict]:
        """Chunk all loaded documents."""
        return self.chunker.chunk_documents(self.documents)

    def retrieve(self, query: str) -> List[dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query

        Returns:
            List of relevant chunks with scores
        """
        if not self.chunks:
            return []

        # Generate query embedding
        query_embedding = self.embedder.encode_single_text(query)

        # Find similar chunks
        similar_chunks = self.embedder.find_similar_texts(
            query_embedding,
            self.chunk_embeddings,
            self.chunk_texts,
            top_k=self.top_k
        )

        # Build result with metadata
        results = []
        for text, score in similar_chunks:
            # Find the original chunk metadata
            for chunk in self.chunks:
                if chunk["text"] == text:
                    result_chunk = chunk.copy()
                    result_chunk["score"] = score
                    results.append(result_chunk)
                    break

        return results

    def generate_response(self, query: str) -> dict:
        """
        Generate a complete response for a query.

        Args:
            query: User's question

        Returns:
            Dictionary with query, response, sources, and metadata
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query)

        # Generate response with sources
        result = self.llm_generator.generate_response_with_sources(
            query=query,
            retrieved_chunks=retrieved_chunks,
            embedder=self.embedder
        )

        # Add pipeline metadata
        result.update({
            "pipeline_info": {
                "total_documents": len(self.documents),
                "total_chunks": len(self.chunks),
                "embedding_model": self.embedder.model_name,
                "llm_model": self.llm_generator.model_name,
                "top_k": self.top_k
            }
        })

        return result

    def run(self, query: str) -> dict:
        """
        Alias for generate_response for backward compatibility.
        """
        return self.generate_response(query)


def create_traditional_rag_pipeline(
    dataset_root: str | Path = "dataset",
    **kwargs
) -> TraditionalRAGPipeline:
    """
    Factory function to create a TraditionalRAGPipeline instance.

    Args:
        dataset_root: Root directory containing documents
        **kwargs: Additional arguments for TraditionalRAGPipeline

    Returns:
        TraditionalRAGPipeline instance
    """
    return TraditionalRAGPipeline(dataset_root=dataset_root, **kwargs)
