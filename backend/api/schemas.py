from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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


class GraphProviderSettingsRequest(BaseModel):
    provider: str


class GraphProviderSettingsResponse(BaseModel):
    provider: str


class GraphProviderOption(BaseModel):
    value: str
    label: str


class GraphProviderOptionsResponse(BaseModel):
    options: List[GraphProviderOption] = Field(default_factory=list)


class GraphProviderSettingsUpdateResponse(BaseModel):
    provider: str
    status: str
    message: str
