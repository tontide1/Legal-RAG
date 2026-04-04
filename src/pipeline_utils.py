from __future__ import annotations

import os


def make_node_id(label: str, name: str) -> str:
    return f"{label}::{name}"


def build_text_payload(name: str, value: object | None) -> str:
    value_text = "" if value is None else str(value)
    return f"{name} {value_text}".strip()


def get_configured_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
