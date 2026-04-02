import time
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from app.schemas.review import (
    EvaluationRequestSchema,
    EvaluationResponseSchema,
    LiveEvaluationRequestSchema,
    LiveEvaluationResponseSchema,
    ReviewRevisionResponseSchema,
)
from app.supervisor.agent import SupervisorAgent
from app.services.workflow import ReviewWorkflowService


router = APIRouter(tags=["evaluation"])

# Dependency to get the supervisor instance
_supervisor: Optional[SupervisorAgent] = None


def get_supervisor() -> SupervisorAgent:
    """Dependency injection for SupervisorAgent"""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _supervisor


def set_supervisor(supervisor: SupervisorAgent):
    """Set the supervisor instance (called from main.py lifespan)"""
    global _supervisor
    _supervisor = supervisor


@router.post("/evaluate", response_model=EvaluationResponseSchema)
async def evaluate_document(
    request: EvaluationRequestSchema,
    supervisor: SupervisorAgent = Depends(get_supervisor)
):
    """
    Evaluate an academic document.
    
    Returns comprehensive evaluation results including:
    - Overall score (0-100)
    - Individual worker scores (grammar, coherence, argumentation, etc.)
    - Recommended actions for improvement
    - Quality control review
    """
    try:
        return await ReviewWorkflowService.evaluate_text(request=request, supervisor=supervisor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/review-revise", response_model=ReviewRevisionResponseSchema)
async def review_and_revise_document(
    request: EvaluationRequestSchema,
    supervisor: SupervisorAgent = Depends(get_supervisor),
):
    """Evaluate a document and return both feedback and a revised draft."""
    try:
        return await ReviewWorkflowService.review_revise_text(
            request=request,
            supervisor=supervisor,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review and revision failed: {str(e)}")


@router.get("/evaluate/{document_id}")
async def get_evaluation_status(document_id: str):
    """Get evaluation status by document ID (placeholder for async evaluations)"""
    return JSONResponse(
        status_code=501,
        content={"detail": "Async evaluation not implemented yet"},
    )


@router.post("/evaluate/live", response_model=LiveEvaluationResponseSchema)
async def evaluate_live(
    request: LiveEvaluationRequestSchema,
    supervisor: SupervisorAgent = Depends(get_supervisor)
):
    """
    Perform a fast, live check on a document fragment.
    
    Triggered by client-side events (pause, sentence completion).
    """
    start_time = time.perf_counter()
    
    try:
        actions = await supervisor.live_check(request.content, request.trigger)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return LiveEvaluationResponseSchema(
            actions=[
                {
                    "type": action.type.value,
                    "target": action.target,
                    "category": action.category,
                    "reasoning": action.reasoning,
                    "suggestion": action.suggestion,
                    "confidence": action.confidence,
                    "highlight": action.highlight.model_dump() if action.highlight else None,
                    "original_text": action.original_text,
                    "corrected_text": action.corrected_text,
                }
                for action in actions
            ],
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live evaluation failed: {str(e)}")

