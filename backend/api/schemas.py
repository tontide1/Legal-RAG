from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    stream: Optional[bool] = False
    comparison_mode: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    mode: str = "hybrid"
    sources: List[Dict[str, Any]] = Field(default_factory=list)

class ComparisonResponse(BaseModel):
    naive: ChatResponse
    hybrid: ChatResponse

class UploadResponse(BaseModel):
    filename: str
    status: str
    message: str
