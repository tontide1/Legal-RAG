from typing import List, Dict
import numpy as np

class SemanticChunker:
    def __init__(self, model):
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for the given text using the specified model."""
        return self.model.encode(text).tolist()

    def chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split the text into chunks based on the specified chunk size."""
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def create_semantic_chunks(self, text: str, chunk_size: int) -> List[Dict[str, List[float]]]:
        """Create semantic chunks from the text."""
        chunks = self.chunk_text(text, chunk_size)
        semantic_chunks = []

        for chunk in chunks:
            embedding = self.embed_text(chunk)
            semantic_chunks.append({'text': chunk, 'embedding': embedding})

        return semantic_chunks

# Example usage:
# model = SomeEmbeddingModel()  # Replace with actual model initialization
# chunker = SemanticChunker(model)
# text = "Your input text goes here."
# chunks = chunker.create_semantic_chunks(text, chunk_size=50)