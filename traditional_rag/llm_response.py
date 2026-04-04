from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .embedding import TextEmbedder

load_dotenv()


class LLMResponseGenerator:
    """Class for generating responses using LLM based on retrieved context."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        """
        Initialize the LLM response generator.

        Args:
            model_name: Name of the Gemini model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )

        # Create the prompt template
        self.prompt_template = PromptTemplate.from_template(
            """Bạn là chuyên gia tư vấn pháp luật tại Việt Nam. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.

Câu hỏi: {query}

Thông tin tham khảo:
{context}

Yêu cầu:
1. Trả lời ngắn gọn, rõ ràng và chính xác
2. Nếu không có thông tin đủ để trả lời, hãy nói rõ
3. Trích dẫn nguồn và điều luật liên quan nếu có
4. Sử dụng ngôn ngữ pháp lý phù hợp

Trả lời:"""
        )

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response based on query and context.

        Args:
            query: User's question
            context: Retrieved context text

        Returns:
            Generated response string
        """
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "query": query,
                "context": context
            })

            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            error_msg = f"Lỗi khi tạo phản hồi: {str(e)}"
            print(error_msg)
            return "Xin lỗi, tôi không thể tạo phản hồi lúc này. Vui lòng thử lại sau."

    def generate_response_with_sources(
        self,
        query: str,
        retrieved_chunks: List[dict],
        embedder: Optional[TextEmbedder] = None
    ) -> dict:
        """
        Generate response with source information.

        Args:
            query: User's question
            retrieved_chunks: List of retrieved text chunks with metadata
            embedder: Optional embedder for reranking

        Returns:
            Dictionary with response and source information
        """
        if not retrieved_chunks:
            return {
                "query": query,
                "response": "Tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này.",
                "sources": [],
                "scores": []
            }

        # Build context from retrieved chunks
        context_parts = []
        sources = []
        scores = []

        for chunk in retrieved_chunks:
            context_parts.append(chunk["text"])
            sources.append({
                "doc_id": chunk.get("doc_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            })
            scores.append(chunk.get("score", 0.0))

        context = "\n\n".join(context_parts)

        # Generate response
        response = self.generate_response(query, context)

        return {
            "query": query,
            "response": response,
            "context": context,
            "sources": sources,
            "scores": scores
        }


def create_response_generator(
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.3
) -> LLMResponseGenerator:
    """
    Factory function to create an LLMResponseGenerator instance.

    Args:
        model_name: Name of the LLM model
        temperature: Temperature for generation

    Returns:
        LLMResponseGenerator instance
    """
    return LLMResponseGenerator(model_name, temperature)