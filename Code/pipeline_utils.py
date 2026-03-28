from __future__ import annotations


def make_node_id(label: str, name: str) -> str:
    return f"{label}::{name}"


def build_text_payload(name: str, value: object | None) -> str:
    value_text = "" if value is None else str(value)
    return f"{name} {value_text}".strip()
