"""
Pydantic Schemas for Request and Response Objects
------------------------------------------------
Enforces strict input validation, response contracts, and structured citations across API v1 endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal


class Citation(BaseModel):
    source_type: str = Field(..., description="Type of source: 'document', 'web', or 'code_execution'")
    filename: str
    chunk_index: int
    line_start: Optional[int] = Field(default=None, description="Actual calculated starting line number in source text")
    line_end: Optional[int] = Field(default=None, description="Actual calculated ending line number in source text")
    snippet: str
    relevance_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or input query", min_length=1)
    max_tokens: Optional[int] = Field(default=150, ge=1, le=1024)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=5, ge=1, le=50)
    engine_mode: Optional[Literal["flash_reasoning", "custom_pytorch_gpt"]] = Field(
        default="flash_reasoning",
        description="Selects primary instruction engine vs custom PyTorch character-level GPT"
    )
    session_id: Optional[str] = Field(default="default_session", description="Client session ID for conversation memory")


class ChatResponse(BaseModel):
    prompt: str
    response: str
    full_text: str
    source_tool: str = "gpt_model"
    engine_mode: str = "flash_reasoning"
    session_id: str = "default_session"
    reasoning_trace: Optional[List[str]] = Field(default_factory=list, description="Agent tool execution trace")
    citations: Optional[List[Citation]] = Field(default_factory=list, description="Structured verified citations")


class DocumentSummaryResponse(BaseModel):
    filename: str
    summary: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    model_loaded: bool
    device: str
