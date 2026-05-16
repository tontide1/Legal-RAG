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


class LocalSentenceTransformerEmbeddingFunc:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.query_instruction = settings.EMBEDDING_QUERY_INSTRUCTION
        self._model = None

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "EMBEDDING_DEVICE is set to CUDA, but the current PyTorch build "
                    "does not have CUDA available. Install a CUDA-enabled PyTorch "
                    "build in the active environment or change EMBEDDING_DEVICE."
                )

            self._model = SentenceTransformer(self.model_name, device=self.device)
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
                batch_size=1,
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
    requested_max_tokens = api_kwargs.get("max_tokens")
    if requested_max_tokens is None:
        api_kwargs["max_tokens"] = settings.LLM_MAX_TOKENS
    else:
        api_kwargs["max_tokens"] = min(int(requested_max_tokens), settings.LLM_MAX_TOKENS)
    
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

async def qwen_vl_parse_pdf(file_path: str) -> str:
    """
    Parse PDF using Qwen 3 VL model via OpenRouter.
    Converts pages to images and sends them to the vision model.
    """
    import base64
    from io import BytesIO
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
    from pypdf import PdfReader

    def extract_text_with_pypdf() -> str:
        reader = PdfReader(file_path)
        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text)
        return "\n\n".join(extracted_pages).strip()
    
    # Convert PDF to images
    # We limit to first few pages for efficiency in this demo, 
    # but you can process all of them.
    try:
        convert_kwargs = {"first_page": 1, "last_page": 5}
        if settings.POPPLER_PATH:
            convert_kwargs["poppler_path"] = settings.POPPLER_PATH
        images = convert_from_path(file_path, **convert_kwargs)
    except PDFInfoNotInstalledError as exc:
        fallback_text = extract_text_with_pypdf()
        if fallback_text:
            return fallback_text
        raise RuntimeError(
            "PDF image parsing requires Poppler. Install Poppler and add its bin folder to PATH, "
            "or set POPPLER_PATH in .env to that bin folder."
        ) from exc
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an elite legal document parser. Extract every character from the images with 100% fidelity. "
                "STRICT INSTRUCTION: Output ONLY the raw extracted text. "
                "NO PREAMBLE. NO CONVERSATION. NO INTRODUCTIONS (e.g., 'Dưới đây là...', 'Here is the text...'). "
                "Directly start with the content of the document. "
                "Maintain original layout, headers, and spacing."
            )
        }
    ]
    
    content = []
    for i, image in enumerate(images):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        content.append({
            "type": "text",
            "text": f"--- Page {i+1} ---"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        })
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    client = get_openai_client()
    response = await client.chat.completions.create(
        model="qwen/qwen3-vl-235b-a22b-instruct",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "https://github.com/traffic/law-assistant",
            "X-Title": "Traffic Law Assistant PDF Parser",
        }
    )
    
    return response.choices[0].message.content
