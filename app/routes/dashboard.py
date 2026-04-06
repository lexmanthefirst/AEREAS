from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.dashboard import (
    ReviewDetailResponseSchema,
    StudentDashboardResponseSchema,
    TeacherDashboardResponseSchema,
)
from app.services.history import HistoryService


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/student/{user_id}", response_model=StudentDashboardResponseSchema)
async def student_dashboard(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    session: AsyncSession = Depends(get_db),
):
    """Student dashboard: review/revision history and score trend snapshot."""
    return await HistoryService.get_student_dashboard(session=session, user_id=user_id, limit=limit)


@router.get("/teacher", response_model=TeacherDashboardResponseSchema)
async def teacher_dashboard(
    teacher_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
):
    """Teacher dashboard: aggregate review metrics and recent review activity."""
    return await HistoryService.get_teacher_dashboard(session=session, teacher_id=teacher_id, limit=limit)


@router.get("/review/{review_id}", response_model=ReviewDetailResponseSchema)
async def review_detail(review_id: UUID, session: AsyncSession = Depends(get_db)):
    """Detailed original/revised content and feedback for a specific review record."""
    data = await HistoryService.get_review_detail(session=session, review_id=review_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Review record not found")
    return data
