from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in os.environ.get("PYTHONPATH", ""):
    import sys
    sys.path.insert(0, str(CODE_ROOT))


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


def run_legal_qa_for_ui(query: str) -> dict:
    from legal_qa import run_legal_qa
    return run_legal_qa(query)
