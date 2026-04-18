from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.context import LiveTriggerType


class EvaluationRequestSchema(BaseModel):
    """Request payload for evaluation and review-revise text endpoints."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., min_length=10, description="Document text to evaluate")
    citation_style: str = Field(default="harvard", description="Citation style (harvard, apa)")
    include_recommendations: bool = Field(default=True)
    requester_id: Optional[str] = Field(default=None, description="Student or teacher identifier")
    requester_role: str = Field(default="student", description="Role: student or teacher")
    use_llm_rewrite: bool = Field(
        default=True,
        description="Enable LLM rewrite pass for critical/moderate issues in review-revise flow",
    )


class EvaluationResponseSchema(BaseModel):
    """Response payload for evaluation endpoint."""

    document_id: str
    overall_score: float
    worker_scores: Dict[str, float]
    actions: List[Dict[str, Any]]
    critic_approved: bool
    synthesis_reasoning: Optional[str] = None
    runtime_info: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float


class ReviewRevisionResponseSchema(BaseModel):
    """Response payload for review/revise endpoints."""

    document_id: str
    overall_score: float
    worker_scores: Dict[str, float]
    actions: List[Dict[str, Any]]
    critic_approved: bool
    synthesis_reasoning: Optional[str] = None
    runtime_info: Dict[str, Any] = Field(default_factory=dict)
    revised_content: str
    revision_mode: str
    rewrite_summary: List[str]
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    tracked_changes: List[Dict[str, Any]] = Field(default_factory=list)
    change_summary: Dict[str, int] = Field(default_factory=dict)
    processing_time_ms: float


class LiveEvaluationRequestSchema(BaseModel):
    """Request payload for live evaluation endpoint."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., description="Current text content")
    trigger: LiveTriggerType = Field(..., description="Event that triggered the check")
    cursor_position: Optional[int] = Field(None, description="Current cursor position")


class LiveEvaluationResponseSchema(BaseModel):
    """Response payload for live evaluation endpoint."""

    actions: List[Dict[str, Any]]
    processing_time_ms: float
