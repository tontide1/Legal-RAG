from typing import Any, Dict, List
import chroma_utils
from llm.gemini import GeminiClient
from llm.openai import OpenAIClient
from llm.ollama import OllamaClient

class Chatbot:
    def __init__(self, llm_choice: str, api_key: str = None):
        self.llm_choice = llm_choice
        self.api_key = api_key
        self.llm_client = self.initialize_llm()

    def initialize_llm(self):
        if self.llm_choice == "gemini":
            return GeminiClient(api_key=self.api_key)
        elif self.llm_choice == "openai":
            return OpenAIClient(api_key=self.api_key)
        elif self.llm_choice == "ollama":
            return OllamaClient()
        else:
            raise ValueError("Invalid LLM choice. Choose 'gemini', 'openai', or 'ollama'.")

    def get_response(self, user_query: str, context: List[str]) -> str:
        embeddings = chroma_utils.embed_texts(context)
        relevant_info = chroma_utils.retrieve_relevant_info(embeddings, user_query)
        response = self.llm_client.generate_response(user_query, relevant_info)
        return response

    def clear_context(self) -> None:
        chroma_utils.clear_embeddings()