from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import backend.core.llm_services as llm_services


def test_build_ollama_index_system_prompt_includes_traffic_law_guidance():
    prompt = llm_services._build_ollama_index_system_prompt("Extract graph records.")

    assert prompt.startswith("TRAFFIC-LAW GRAPH EXTRACTION")
    assert "Preserve exact Vietnamese legal phrases" in prompt
    assert "Do not paraphrase or normalize legal citations" in prompt
    assert "Văn bản pháp luật" in prompt
    assert "Điều khoản" in prompt
    assert "Hành vi vi phạm" in prompt
    assert "Giấy phép / chứng chỉ" in prompt
    assert "STRICT OUTPUT RULES:" in prompt
    assert "Extract graph records." in prompt
