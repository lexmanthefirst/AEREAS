import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
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


def get_supervisor(request: Request) -> SupervisorAgent:
    """Dependency injection for SupervisorAgent via app.state."""
    supervisor: SupervisorAgent | None = getattr(request.app.state, "supervisor", None)
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return supervisor


@router.post("/evaluate", response_model=EvaluationResponseSchema)
async def evaluate_document(
    request: EvaluationRequestSchema,
    supervisor: SupervisorAgent = Depends(get_supervisor),
):
    """Evaluate an academic document."""
    return await ReviewWorkflowService.evaluate_text(request=request, supervisor=supervisor)


@router.post("/review-revise", response_model=ReviewRevisionResponseSchema)
async def review_and_revise_document(
    request: EvaluationRequestSchema,
    supervisor: SupervisorAgent = Depends(get_supervisor),
    session: AsyncSession = Depends(get_db),
):
    """Evaluate a document and return both feedback and a revised draft."""
    return await ReviewWorkflowService.review_revise_text(
        request=request,
        supervisor=supervisor,
        session=session,
    )


@router.get("/evaluate/{document_id}")
async def get_evaluation_status(document_id: str):
    """Get evaluation status by document ID (placeholder for async evaluations)."""
    return JSONResponse(
        status_code=501,
        content={"detail": "Async evaluation not implemented yet"},
    )


@router.post("/evaluate/live", response_model=LiveEvaluationResponseSchema)
async def evaluate_live(
    request: LiveEvaluationRequestSchema,
    supervisor: SupervisorAgent = Depends(get_supervisor),
):
    """Perform a fast, live check on a document fragment."""
    start_time = time.perf_counter()

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
