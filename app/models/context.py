from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    """Types of recommended actions"""
    CRITICAL_REVISION = "critical_revision"  # Severe issues requiring rewrite
    MODERATE_REVISION = "moderate_revision"  # Notable issues to address
    MINOR_IMPROVEMENT = "minor_improvement"  # Polish suggestions
    POSITIVE_FEEDBACK = "positive_feedback"  # Highlight strengths


class LiveTriggerType(str, Enum):
    """Trigger type for live evaluation"""
    PAUSE = "pause"          # User paused typing
    SENTENCE = "sentence"    # User completed a sentence
    PARAGRAPH = "paragraph"  # User completed a paragraph


class TextSpan(BaseModel):
    """Text position for frontend highlighting"""
    start: int              # Character start position (0-indexed)
    end: int                # Character end position (exclusive)
    text: str = ""          # The actual text content


class EvaluationAction(BaseModel):
    """Proposed action from worker or supervisor synthesis"""
    type: ActionType
    target: str                   # Sentence/paragraph/section identifier
    category: str                 # "grammar", "coherence", etc.
    reasoning: str                # Explanation for the action
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    suggestion: Optional[str] = None  # Specific improvement suggestion
    
    # For frontend highlighting
    highlight: Optional[TextSpan] = None  # Position in original document
    original_text: Optional[str] = None   # The problematic text
    corrected_text: Optional[str] = None  # Suggested correction
    
    details: Dict[str, Any] = Field(default_factory=dict)


class WorkerResult(BaseModel):
    """Specialist worker evaluation report posted to the board"""
    worker_name: str = ""                    # "grammar_specialist", etc.
    score: float = Field(ge=0.0, le=100.0, default=0.0)
    findings: List[str] = Field(default_factory=list)  # Human-readable findings
    flagged_items: List[Dict[str, Any]] = Field(default_factory=list)  # Specific issues
    proposed_actions: List[EvaluationAction] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SynthesisResult(BaseModel):
    """Result from the SynthesisEngine"""
    reasoning: str = ""
    actions: List[EvaluationAction] = Field(default_factory=list)


class CriticReview(BaseModel):
    """Quality control review from CriticWorker"""
    approved: bool = True
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationContext(BaseModel):
    """
    The Evaluation Board (Blackboard) - shared memory between all agents.
    
    Workers READ from it (document content) and WRITE to it (their findings).
    No direct worker-to-worker communication - all goes through this board.
    """
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID = Field(default_factory=uuid4)
    
    document_content: str = ""
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    worker_results: Dict[str, WorkerResult] = Field(default_factory=dict)
    synthesis_reasoning: Optional[str] = None
    final_actions: List[EvaluationAction] = Field(default_factory=list)
    critic_review: Optional[CriticReview] = None
    final_scores: Dict[str, float] = Field(default_factory=dict)
    overall_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed, failed

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


