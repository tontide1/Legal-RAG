import os
import openai
import asyncio
from typing import List, Union, Optional
from backend.config import settings

# Global client cache to avoid pickling issues and redundant connections
_async_client: Optional[openai.AsyncOpenAI] = None

def get_openai_client():
    global _async_client
    if _async_client is None:
        _async_client = openai.AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    return _async_client

class QwenEmbeddingFunc:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL

    def _get_prefix(self, is_query: bool) -> str:
        if is_query:
            return "Instruct: Given a legal query, retrieve relevant statutes...\nQuery: "
        return ""

    async def __call__(self, texts: List[str]):
        import numpy as np
        client = get_openai_client()
        results = []
        for text in texts:
            prefix = self._get_prefix(is_query=text.strip().endswith("?"))
            
            response = await client.embeddings.create(
                model=self.model_name,
                input=prefix + text
            )
            results.append(response.data[0].embedding)
        return np.array(results)


class VietLegalHarrierEmbeddingFunc:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.query_instruction = settings.EMBEDDING_QUERY_INSTRUCTION
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _is_query(self, text: str) -> bool:
        stripped = text.strip()
        return stripped.endswith("?") or stripped.lower().startswith(("ai ", "là gì", "thế nào", "khi nào", "bao giờ"))

    def _prepare_text(self, text: str) -> str:
        if self._is_query(text):
            return f"{self.query_instruction}{text}"
        return text

    async def __call__(self, texts: List[str]):
        import numpy as np

        prepared_texts = [self._prepare_text(text) for text in texts]

        def encode_batch():
            model = self._get_model()
            return model.encode(
                prepared_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        embeddings = await asyncio.to_thread(encode_batch)
        return np.array(embeddings)

# OpenRouter LLM Wrapper for LightRAG
async def deepseek_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs
) -> str:
    client = get_openai_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if history:
        messages.extend(history)
        
    messages.append({"role": "user", "content": prompt})
    
    # OpenRouter specific headers
    extra_headers = {
        "HTTP-Referer": "https://github.com/traffic/law-assistant",
        "X-Title": "Traffic Law Assistant",
    }
    
    # Filter out kwargs that are for LightRAG internal use or not accepted by OpenAI
    # LightRAG sometimes passes 'hashing_kv' or other parameters for its internal caching logic
    allowed_params = [
        "model", "messages", "stream", "temperature", "top_p", "n", "stop", "max_tokens",
        "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format",
        "seed", "tools", "tool_choice", "parallel_tool_calls"
    ]
    api_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
    api_kwargs.setdefault("temperature", 0.3)
    api_kwargs.setdefault("n", 1)
    
    response = await client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        extra_headers=extra_headers,
        **api_kwargs
    )
    
    if api_kwargs.get("stream"):
        async def stream_generator():
            print("LLM: Starting stream generator")
            try:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        c = chunk.choices[0].delta.content
                        print(f"LLM CHUNK: {c}")
                        yield c
            except Exception as e:
                print(f"LLM STREAM ERROR: {str(e)}")
            print("LLM: Stream generator finished")
        return stream_generator()
    else:
        return response.choices[0].message.content
