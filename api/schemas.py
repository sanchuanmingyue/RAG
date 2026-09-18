"""HTTP request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RetrievalMode = Literal["vector", "keyword", "hybrid"]


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    paper_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_mode: RetrievalMode = "hybrid"
    expand_parent: bool | None = None


class ChatRequest(RetrieveRequest):
    pass


class SummaryRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=12, ge=1, le=50)


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    current_paper_id: str | None = Field(default=None, max_length=500)
    selected_paper_ids: list[str] = Field(default_factory=list, max_length=50)


class LiteratureSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5_000)
    limit: int = Field(default=10, ge=1, le=50)


class TaskResponse(BaseModel):
    task_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    progress_stage: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: float = 0.0
    progress_message: str | None = None


class MessageResponse(BaseModel):
    message: str
