from __future__ import annotations

import os


def judge_enabled() -> bool:
    return os.getenv("EVAL_ENABLE_LLM_JUDGE", "true").lower() == "true"


def get_judge_model() -> str:
    return os.getenv("EVAL_JUDGE_MODEL", "gemini-2.5-flash-lite")


def run_judge_metrics(*, enabled: bool | None = None) -> dict:
    is_enabled = judge_enabled() if enabled is None else enabled
    if not is_enabled:
        return {"judge_metrics_skipped": True, "reason": "LLM judge disabled"}
    return {"judge_metrics_skipped": True, "reason": f"Judge placeholder for model {get_judge_model()}"}
