from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, TypeVar

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - fallback for non-UI test envs
    st = None

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in os.environ.get("PYTHONPATH", ""):
    import sys
    sys.path.insert(0, str(CODE_ROOT))


F = TypeVar("F", bound=Callable)


def _cache_resource(func: F) -> F:
    if st is None:
        return func
    return st.cache_resource(show_spinner=False)(func)


def check_env_status() -> dict[str, str]:
    load_dotenv()
    status = {}
    status["NEO4J_URI"] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    status["NEO4J_USER"] = os.getenv("NEO4J_USER", "neo4j")
    status["NEO4J_PASSWORD"] = "SET" if os.getenv("NEO4J_PASSWORD") else "MISSING"
    status["GOOGLE_API_KEY"] = "SET" if os.getenv("GOOGLE_API_KEY") else "MISSING"
    status["GEMINI_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    status["NER_BACKEND"] = os.getenv("NER_BACKEND", "phobert")
    return status


@_cache_resource
def _get_cached_retriever():
    from retrive.multi_retr import Retrive

    load_dotenv()
    return Retrive(verbose=False)


def _retrieve_with_cached_runtime(query: str, ner_entities: list[str]) -> list[dict]:
    retriever = _get_cached_retriever()
    return retriever.advanced_retrieve(query, ner_entities)


@_cache_resource
def _get_cached_answer_chain():
    from legal_qa import build_answer_chain

    load_dotenv()
    return build_answer_chain()


def _generate_with_cached_runtime(query: str, context_text: str) -> str:
    response = _get_cached_answer_chain().invoke(
        {"query": query, "source_information": context_text}
    )
    return response.content if hasattr(response, "content") else str(response)


def run_legal_qa_for_ui(query: str) -> dict:
    from legal_qa import run_legal_qa

    return run_legal_qa(
        query,
        retrieve_fn=_retrieve_with_cached_runtime,
        generate_answer_fn=_generate_with_cached_runtime,
    )
