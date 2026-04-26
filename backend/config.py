from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import json
from pydantic import field_validator

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
    
    EMBEDDING_BACKEND: str = "openrouter"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    EMBEDDING_DIM: int = 1536
    EMBEDDING_QUERY_INSTRUCTION: str = (
        "Instruct: Given a Vietnamese legal question, retrieve relevant legal passages "
        "that answer the question\nQuery: "
    )
    LLM_MODEL: str = "gemini-3-flash-preview"
    
    SUMMARY_LANGUAGE: str = "Vietnamese"
    ENTITY_TYPES: list[str] = [
        "Văn bản pháp luật", "Điều khoản", "Cơ quan ban hành", "Đối tượng áp dụng", 
        "Hành vi vi phạm", "Hình thức xử phạt", "Thời hạn", "Khái niệm pháp lý"
    ]
    
    LIGHTRAG_WORKING_DIR: str = "./backend/data"

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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
