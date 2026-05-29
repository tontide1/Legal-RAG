from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import json
from pydantic import field_validator

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/law_assistant"
    
    # Postgres individual components for LightRAG
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DATABASE: str = "law_assistant"
    
    REDIS_URL: Optional[str] = None
    
    OPENROUTER_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    POPPLER_PATH: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_INDEX_MODEL: str = "qwen2.5:3b"
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_TIMEOUT_SECONDS: int = 180
    OLLAMA_MAX_RETRIES: int = 2
    OLLAMA_RETRY_DELAY_SECONDS: int = 5
    
    EMBEDDING_BACKEND: str = "sentence_transformers"
    EMBEDDING_MODEL: str = "huyydangg/DEk21_hcmute_embedding"
    EMBEDDING_DIM: int = 768
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_MAX_TOKEN_SIZE: int = 384
    EMBEDDING_QUERY_INSTRUCTION: str = (
        "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages "
        "that answer the question\nQuery: "
    )
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    LLM_MAX_TOKENS: int = 1024
    LIGHTRAG_MAX_ASYNC: int = 1
    LIGHTRAG_EMBEDDING_MAX_ASYNC: int = 2
    LIGHTRAG_EMBEDDING_TIMEOUT: int = 180
    LIGHTRAG_MAX_PARALLEL_INSERT: int = 1
    LIGHTRAG_CHUNK_SIZE: int = 600
    LIGHTRAG_CHUNK_OVERLAP_SIZE: int = 100
    GEMINI_MAX_RETRIES: int = 6
    HYBRID_MAX_HISTORY_MESSAGES: int = 8
    HYBRID_TOP_K: int = 20
    HYBRID_CHUNK_TOP_K: int = 12
    HYBRID_ANCHOR_CHUNK_LIMIT: int = 3
    HYBRID_BUCKET_CHUNK_LIMIT: int = 2
    
    SUMMARY_LANGUAGE: str = "Vietnamese"
    ENTITY_TYPES: list[str] = [
        "Văn bản pháp luật",
        "Điều khoản",
        "Cơ quan ban hành",
        "Đối tượng áp dụng",
        "Thời hạn",
        "Khái niệm pháp lý",
        "Phạm vi áp dụng",
        "Trách nhiệm",
        "Ngoại lệ",
        "Điều kiện áp dụng",
        "Chủ thể có thẩm quyền",
        "Phương tiện giao thông",
        "Người tham gia giao thông",
        "Hành vi bị cấm",
        "Yêu cầu an toàn",
        "Giấy phép / chứng chỉ",
        "Dịch vụ hỗ trợ giao thông",
        "Kết cấu hạ tầng giao thông",
        "Hình thức xử phạt",
        "Hành vi vi phạm",
    ]
    
    LIGHTRAG_WORKING_DIR: str = "./backend/data"

    # Path to Poppler bin directory (required on Windows, e.g. "poppler-24.11.0\Library\bin")
    POPPLER_PATH: Optional[str] = None

    @field_validator("ENTITY_TYPES", mode="before")
    @classmethod
    def parse_entity_types(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except:
                    pass
            return [x.strip() for x in v.split(",")]
        return v

    @field_validator("EMBEDDING_QUERY_INSTRUCTION", mode="before")
    @classmethod
    def normalize_embedding_query_instruction(cls, v):
        if isinstance(v, str):
            return v.replace("\\n", "\n")
        return v

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

settings = Settings()
