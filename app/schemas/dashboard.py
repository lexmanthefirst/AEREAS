from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StudentReviewItemSchema(BaseModel):
    document_id: str
    review_id: str
    revision_id: str
    source_filename: Optional[str] = None
    overall_score: float
    revision_mode: str
    critic_approved: bool
    change_summary: Dict[str, int]
    created_at: str


class StudentDashboardResponseSchema(BaseModel):
    user_id: str
    total_reviews: int
    average_score: float
    recent_reviews: List[StudentReviewItemSchema]


class TeacherReviewItemSchema(BaseModel):
    document_id: str
    review_id: str
    owner_id: Optional[str] = None
    owner_role: str
    source_filename: Optional[str] = None
    overall_score: float
    critic_approved: bool
    revision_mode: str
    created_at: str


class TeacherDashboardResponseSchema(BaseModel):
    teacher_id: Optional[str] = None
    total_reviews: int
    average_score: float
    needs_attention_count: int
    recent_reviews: List[TeacherReviewItemSchema]


class ReviewDetailResponseSchema(BaseModel):
    document_id: str
    review_id: str
    revision_id: str
    owner_id: Optional[str] = None
    owner_role: str
    source_filename: Optional[str] = None
    original_content: str
    revised_content: str
    overall_score: float
    worker_scores: Dict[str, float]
    actions: List[Dict[str, Any]]
    critic_approved: bool
    revision_mode: str
    rewrite_summary: List[str]
    quality_metrics: Dict[str, Any]
    tracked_changes: List[Dict[str, Any]]
    change_summary: Dict[str, int]
    created_at: str
