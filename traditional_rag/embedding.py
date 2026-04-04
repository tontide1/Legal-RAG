from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


class TextEmbedder:
    """Class for generating embeddings from text using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "keepitreal/vietnamese-sbert",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the text embedder.

        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu', 'cuda', etc.)
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir

        print(f"Loading embedding model: {model_name} on device: {self.device}")
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=self.cache_dir
        )

    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def encode_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to encode

        Returns:
            Numpy array embedding
        """
        return self.encode_texts([text], batch_size=1, show_progress=False)[0]

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        return float(np.dot(embedding1, embedding2))

    def find_similar_texts(
        self,
        query_embedding: np.ndarray,
        text_embeddings: np.ndarray,
        texts: List[str],
        top_k: int = 5
    ) -> List[tuple[str, float]]:
        """
        Find most similar texts to a query embedding.

        Args:
            query_embedding: Query embedding
            text_embeddings: Array of text embeddings to search
            texts: Corresponding text strings
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        if len(text_embeddings) == 0:
            return []

        # Calculate similarities
        similarities = np.dot(text_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((texts[idx], float(similarities[idx])))

        return results


def create_embedder(model_name: str = "keepitreal/vietnamese-sbert") -> TextEmbedder:
    """
    Factory function to create a TextEmbedder instance.

    Args:
        model_name: Name of the embedding model

    Returns:
        TextEmbedder instance
    """
    return TextEmbedder(model_name)