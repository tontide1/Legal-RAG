import asyncio
import openai
import re
from typing import List, Optional
from google import genai
from google.genai import types
from lightrag.rerank import jina_rerank
from ollama import AsyncClient as OllamaAsyncClient
from backend.config import settings

# Global client caches to avoid redundant connections
_openrouter_client: Optional[openai.AsyncOpenAI] = None
_gemini_client: Optional[genai.Client] = None
_ollama_client: Optional[OllamaAsyncClient] = None
_nine_router_client: Optional[openai.AsyncOpenAI] = None
_gemini_request_semaphore = asyncio.Semaphore(1)

def get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = openai.AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    return _openrouter_client

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not settings.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not configured. Set GOOGLE_API_KEY in .env "
                "to enable Gemini chat generation."
            )
        _gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _gemini_client


def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaAsyncClient(host=settings.OLLAMA_BASE_URL)
    return _ollama_client


def get_nine_router_client():
    global _nine_router_client
    if _nine_router_client is None:
        api_key = (settings.NINE_ROUTER_API_KEY or "").strip()
        if not api_key:
            raise RuntimeError(
                "NINE_ROUTER_API_KEY is not configured. Set NINE_ROUTER_API_KEY in .env "
                "to enable 9router indexing."
            )
        _nine_router_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=settings.NINE_ROUTER_BASE_URL,
        )
    return _nine_router_client


def hybrid_rerank_available() -> bool:
    return bool(settings.HYBRID_ENABLE_RERANK and (settings.JINA_API_KEY or "").strip())

class QwenEmbeddingFunc:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL

    def _get_prefix(self, is_query: bool) -> str:
        if is_query:
            return "Instruct: Given a legal query, retrieve relevant statutes...\nQuery: "
        return ""

    async def __call__(self, texts: List[str]):
        import numpy as np
        client = get_openrouter_client()
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


def _build_gemini_prompt(
    prompt: str,
    system_prompt: str | None = None,
    history: List[dict] | None = None,
) -> str:
    sections: list[str] = []
    if system_prompt:
        sections.append(f"[System]\n{system_prompt.strip()}")

    if history:
        history_lines: list[str] = []
        for item in history:
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if content:
                history_lines.append(f"{role.title()}: {content}")
        if history_lines:
            sections.append("[History]\n" + "\n".join(history_lines))

    sections.append(f"[User]\n{prompt.strip()}")
    return "\n\n".join(sections)


def _build_ollama_index_system_prompt(system_prompt: str | None) -> str:
    traffic_law_guidance = _build_traffic_law_extraction_guidance()
    strict_format_prompt = (
        "STRICT OUTPUT RULES:\n"
        "- Return only records in the exact extraction format requested.\n"
        "- Do not add explanations, markdown, comments, or code fences.\n"
        "- Do not invent extra columns or fields.\n"
        "- If a field contains punctuation, keep it inside the field text instead of splitting it.\n"
        "- If unsure, return fewer well-formed records rather than malformed output."
    )

    sections = [traffic_law_guidance]
    if system_prompt and system_prompt.strip():
        sections.append(system_prompt.strip())
    sections.append(strict_format_prompt)
    return "\n\n".join(sections)


def _build_openai_index_messages(
    prompt: str,
    system_prompt: str | None = None,
    history: List[dict] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    if history:
        for item in history:
            role = str(item.get("role", "user")).strip() or "user"
            content = str(item.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def _build_traffic_law_extraction_guidance() -> str:
    entity_types = "\n".join(f"- {entity_type}" for entity_type in settings.ENTITY_TYPES)
    return (
        "TRAFFIC-LAW GRAPH EXTRACTION:\n"
        "- Preserve exact Vietnamese legal phrases, article labels, clause references, and sanction terms.\n"
        "- Do not paraphrase or normalize legal citations, named authorities, subject classes, or violation names.\n"
        "- Extract only entities and relations grounded in the provided text.\n"
        "- Prefer the most specific traffic-law entity type that matches the source span.\n"
        "Entity types:\n"
        f"{entity_types}"
    )


def _looks_malformed_extraction_output(response_text: str) -> bool:
    lowered = response_text.lower()
    malformed_markers = (
        "```",
        "here is",
        "dưới đây",
        "giải thích",
        "explanation",
    )
    return any(marker in lowered for marker in malformed_markers)


def _extract_retry_delay_seconds(error: Exception) -> float:
    message = str(error)
    retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if retry_match:
        return max(float(retry_match.group(1)), 1.0)
    return 15.0


def _is_gemini_rate_limit_error(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message


async def _gemini_generate_with_retry(client: genai.Client, **request_kwargs):
    last_error: Exception | None = None

    for attempt in range(1, settings.GEMINI_MAX_RETRIES + 1):
        try:
            async with _gemini_request_semaphore:
                return await client.aio.models.generate_content(**request_kwargs)
        except Exception as error:
            last_error = error
            if not _is_gemini_rate_limit_error(error) or attempt >= settings.GEMINI_MAX_RETRIES:
                raise

            delay_seconds = _extract_retry_delay_seconds(error)
            print(
                f"LLM RATE LIMIT: Gemini quota hit on attempt {attempt}/"
                f"{settings.GEMINI_MAX_RETRIES}. Waiting {delay_seconds:.1f}s before retry."
            )
            await asyncio.sleep(delay_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini request failed without returning a response.")

async def gemini_chat_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs
) -> str:
    client = get_gemini_client()
    request_text = _build_gemini_prompt(
        prompt,
        system_prompt=system_prompt,
        history=history,
    )

    requested_max_tokens = kwargs.get("max_tokens")
    if requested_max_tokens is None:
        max_output_tokens = settings.LLM_MAX_TOKENS
    else:
        max_output_tokens = min(int(requested_max_tokens), settings.LLM_MAX_TOKENS)

    generation_config = types.GenerateContentConfig(
        temperature=float(kwargs.get("temperature", 0.3)),
        top_p=kwargs.get("top_p"),
        max_output_tokens=max_output_tokens,
        stop_sequences=kwargs.get("stop"),
    )

    if kwargs.get("stream"):
        async def stream_generator():
            print("LLM: Starting Gemini stream generator")
            try:
                async with _gemini_request_semaphore:
                    stream = await client.aio.models.generate_content_stream(
                        model=settings.LLM_MODEL,
                        contents=request_text,
                        config=generation_config,
                    )
                async for chunk in stream:
                    c = getattr(chunk, "text", None)
                    if c:
                        print(f"LLM CHUNK: {c}")
                        yield c
            except Exception as e:
                print(f"LLM STREAM ERROR: {str(e)}")
            print("LLM: Gemini stream generator finished")
        return stream_generator()

    response = await _gemini_generate_with_retry(
        client,
        model=settings.LLM_MODEL,
        contents=request_text,
        config=generation_config,
    )
    return response.text or ""


async def ollama_index_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs
) -> str:
    client = get_ollama_client()
    request_text = _build_gemini_prompt(
        prompt,
        system_prompt=_build_ollama_index_system_prompt(system_prompt),
        history=history,
    )
    requested_max_tokens = kwargs.get("max_tokens")
    max_output_tokens = settings.LLM_MAX_TOKENS if requested_max_tokens is None else min(
        int(requested_max_tokens), settings.LLM_MAX_TOKENS
    )

    last_error: Exception | None = None
    retry_prompt = request_text

    for attempt in range(1, settings.OLLAMA_MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.generate(
                    model=settings.OLLAMA_INDEX_MODEL,
                    prompt=retry_prompt,
                    options={
                        "temperature": float(kwargs.get("temperature", 0.0)),
                        "num_ctx": settings.OLLAMA_NUM_CTX,
                        "num_predict": max_output_tokens,
                    },
                    stream=False,
                ),
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
            )
            response_text = str(response.get("response", "")).strip()

            if response_text and _looks_malformed_extraction_output(response_text):
                print(
                    f"LLM WARNING: Ollama extraction output looked malformed on attempt "
                    f"{attempt}/{settings.OLLAMA_MAX_RETRIES}. Retrying with stricter reminder."
                )
                if attempt >= settings.OLLAMA_MAX_RETRIES:
                    return response_text
                retry_prompt = (
                    f"{request_text}\n\n"
                    "FINAL REMINDER: return only exact extraction records with no prose and no extra fields."
                )
                await asyncio.sleep(settings.OLLAMA_RETRY_DELAY_SECONDS)
                continue

            return response_text
        except asyncio.TimeoutError as error:
            last_error = error
            print(
                f"LLM TIMEOUT: Ollama request exceeded {settings.OLLAMA_TIMEOUT_SECONDS}s "
                f"on attempt {attempt}/{settings.OLLAMA_MAX_RETRIES}."
            )
        except Exception as error:
            last_error = error
            print(
                f"LLM ERROR: Ollama request failed on attempt {attempt}/"
                f"{settings.OLLAMA_MAX_RETRIES}: {error}"
            )

        if attempt < settings.OLLAMA_MAX_RETRIES:
            await asyncio.sleep(settings.OLLAMA_RETRY_DELAY_SECONDS)

    if last_error is not None:
        raise RuntimeError(
            "Ollama indexing request failed after retries. "
            f"Last error: {last_error}"
        ) from last_error
    raise RuntimeError("Ollama indexing request failed without a response.")


async def validate_nine_router_connection() -> None:
    client = get_nine_router_client()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.NINE_ROUTER_INDEX_MODEL,
                messages=[{"role": "user", "content": "Respond with OK"}],
                temperature=0.0,
                max_tokens=8,
            ),
            timeout=min(settings.NINE_ROUTER_TIMEOUT_SECONDS, 15),
        )
    except Exception as error:
        raise RuntimeError(
            f"9router local proxy is not reachable at {settings.NINE_ROUTER_BASE_URL}: {error}"
        ) from error

    response_text = (
        response.choices[0].message.content
        if response.choices and response.choices[0].message
        else ""
    )
    if not str(response_text).strip():
        raise RuntimeError(
            f"9router local proxy returned an empty validation response at "
            f"{settings.NINE_ROUTER_BASE_URL}"
        )


async def nine_router_index_llm_func(
    prompt: str,
    system_prompt: str = None,
    history: List[dict] = None,
    **kwargs
) -> str:
    client = get_nine_router_client()
    messages = _build_openai_index_messages(
        prompt,
        system_prompt=_build_ollama_index_system_prompt(system_prompt),
        history=history,
    )
    requested_max_tokens = kwargs.get("max_tokens")
    max_output_tokens = settings.LLM_MAX_TOKENS if requested_max_tokens is None else min(
        int(requested_max_tokens), settings.LLM_MAX_TOKENS
    )

    last_error: Exception | None = None
    retry_messages = messages

    for attempt in range(1, settings.NINE_ROUTER_MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=settings.NINE_ROUTER_INDEX_MODEL,
                    messages=retry_messages,
                    temperature=float(kwargs.get("temperature", 0.0)),
                    max_tokens=max_output_tokens,
                ),
                timeout=settings.NINE_ROUTER_TIMEOUT_SECONDS,
            )
            response_text = ""
            if response.choices and response.choices[0].message:
                response_text = str(response.choices[0].message.content or "").strip()

            if response_text and _looks_malformed_extraction_output(response_text):
                print(
                    f"LLM WARNING: 9router extraction output looked malformed on attempt "
                    f"{attempt}/{settings.NINE_ROUTER_MAX_RETRIES}. Retrying with stricter reminder."
                )
                if attempt >= settings.NINE_ROUTER_MAX_RETRIES:
                    return response_text
                retry_messages = retry_messages + [{
                    "role": "user",
                    "content": (
                        "FINAL REMINDER: return only exact extraction records with no prose and no extra fields."
                    ),
                }]
                await asyncio.sleep(settings.NINE_ROUTER_RETRY_DELAY_SECONDS)
                continue

            return response_text
        except asyncio.TimeoutError as error:
            last_error = error
            print(
                f"LLM TIMEOUT: 9router request exceeded {settings.NINE_ROUTER_TIMEOUT_SECONDS}s "
                f"on attempt {attempt}/{settings.NINE_ROUTER_MAX_RETRIES}."
            )
        except Exception as error:
            last_error = error
            print(
                f"LLM ERROR: 9router request failed on attempt {attempt}/"
                f"{settings.NINE_ROUTER_MAX_RETRIES}: {error}"
            )

        if attempt < settings.NINE_ROUTER_MAX_RETRIES:
            await asyncio.sleep(settings.NINE_ROUTER_RETRY_DELAY_SECONDS)

    if last_error is not None:
        raise RuntimeError(
            "9router indexing request failed after retries. "
            f"Last error: {last_error}"
        ) from last_error
    raise RuntimeError("9router indexing request failed without a response.")


async def jina_rerank_model_func(
    query: str,
    documents: List[str],
    top_n: int | None = None,
):
    return await jina_rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        api_key=settings.JINA_API_KEY,
        model=settings.JINA_RERANK_MODEL,
        base_url=settings.JINA_RERANK_BASE_URL,
    )


# Backward-compatible alias for existing imports/tests.
deepseek_llm_func = gemini_chat_llm_func
