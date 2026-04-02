from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import desc, func, select

from app.db.models import DocumentRecord, ReviewRecord, RevisionRecord
from app.db.session import SessionLocal
from app.models.context import EvaluationAction, EvaluationContext


def _serialize_actions(actions: List[EvaluationAction]) -> List[Dict[str, Any]]:
    return [action.model_dump(mode="json") for action in actions]


class HistoryService:
    """Persist and query review/revision history for dashboards."""

    @staticmethod
    def _base_review_join_query():
        return (
            select(DocumentRecord, ReviewRecord, RevisionRecord)
            .join(ReviewRecord, ReviewRecord.document_id == DocumentRecord.id)
            .join(RevisionRecord, RevisionRecord.review_id == ReviewRecord.id)
        )

    @staticmethod
    async def _score_summary(session, owner_id: Optional[str] = None) -> tuple[int, float]:
        stmt = select(func.count(ReviewRecord.id), func.avg(ReviewRecord.overall_score)).join(
            DocumentRecord, ReviewRecord.document_id == DocumentRecord.id
        )
        if owner_id:
            stmt = stmt.where(DocumentRecord.owner_id == owner_id)

        total_reviews, avg_score = (await session.execute(stmt)).one()
        return int(total_reviews or 0), round(float(avg_score or 0.0), 2)

    @staticmethod
    def _student_dashboard_row(
        doc: DocumentRecord,
        review: ReviewRecord,
        revision: RevisionRecord,
    ) -> Dict[str, Any]:
        return {
            "document_id": str(doc.id),
            "review_id": str(review.id),
            "revision_id": str(revision.id),
            "source_filename": doc.source_filename,
            "overall_score": review.overall_score,
            "revision_mode": revision.revision_mode,
            "critic_approved": review.critic_approved,
            "change_summary": revision.change_summary,
            "created_at": review.created_at.isoformat(),
        }

    @staticmethod
    def _teacher_dashboard_row(
        doc: DocumentRecord,
        review: ReviewRecord,
        revision: RevisionRecord,
    ) -> Dict[str, Any]:
        return {
            "document_id": str(doc.id),
            "review_id": str(review.id),
            "owner_id": doc.owner_id,
            "owner_role": doc.owner_role,
            "source_filename": doc.source_filename,
            "overall_score": review.overall_score,
            "critic_approved": review.critic_approved,
            "revision_mode": revision.revision_mode,
            "created_at": review.created_at.isoformat(),
        }

    @staticmethod
    def _review_detail_payload(
        doc: DocumentRecord,
        review: ReviewRecord,
        revision: RevisionRecord,
    ) -> Dict[str, Any]:
        return {
            "document_id": str(doc.id),
            "review_id": str(review.id),
            "revision_id": str(revision.id),
            "owner_id": doc.owner_id,
            "owner_role": doc.owner_role,
            "source_filename": doc.source_filename,
            "original_content": doc.original_content,
            "revised_content": revision.revised_content,
            "overall_score": review.overall_score,
            "worker_scores": review.worker_scores,
            "actions": review.actions,
            "critic_approved": review.critic_approved,
            "revision_mode": revision.revision_mode,
            "rewrite_summary": revision.rewrite_summary,
            "quality_metrics": revision.quality_metrics,
            "tracked_changes": revision.tracked_changes,
            "change_summary": revision.change_summary,
            "created_at": review.created_at.isoformat(),
        }

    @staticmethod
    async def persist_review_revision(
        *,
        context: EvaluationContext,
        revision_result: Dict[str, Any],
        processing_time_ms: float,
        requester_id: Optional[str],
        requester_role: str,
        source_filename: Optional[str] = None,
    ) -> Dict[str, str]:
        async with SessionLocal() as session:
            document = DocumentRecord(
                owner_id=requester_id,
                owner_role=requester_role,
                source_filename=source_filename,
                original_content=context.document_content,
            )
            await document.insert(session, flush=True)

            review = ReviewRecord(
                document_id=document.id,
                overall_score=context.overall_score or 0.0,
                worker_scores=context.final_scores,
                actions=_serialize_actions(context.final_actions),
                critic_approved=context.critic_review.approved if context.critic_review else True,
                processing_time_ms=processing_time_ms,
            )
            await review.insert(session, flush=True)

            revision = RevisionRecord(
                review_id=review.id,
                revised_content=revision_result["revised_content"],
                revision_mode=revision_result["revision_mode"],
                rewrite_summary=revision_result["rewrite_summary"],
                quality_metrics=revision_result["quality_metrics"],
                tracked_changes=revision_result["tracked_changes"],
                change_summary=revision_result["change_summary"],
            )
            await revision.insert(session, flush=False)

            await session.commit()

            return {
                "document_id": str(document.id),
                "review_id": str(review.id),
                "revision_id": str(revision.id),
            }

    @staticmethod
    async def get_student_dashboard(user_id: str, limit: int = 20) -> Dict[str, Any]:
        async with SessionLocal() as session:
            stmt = (
                HistoryService._base_review_join_query()
                .where(DocumentRecord.owner_id == user_id)
                .order_by(desc(ReviewRecord.created_at))
                .limit(limit)
            )
            rows = (await session.execute(stmt)).all()
            total_reviews, avg_score = await HistoryService._score_summary(session, owner_id=user_id)

            items = [
                HistoryService._student_dashboard_row(doc, review, revision)
                for doc, review, revision in rows
            ]

            return {
                "user_id": user_id,
                "total_reviews": total_reviews,
                "average_score": avg_score,
                "recent_reviews": items,
            }

    @staticmethod
    async def get_teacher_dashboard(teacher_id: Optional[str], limit: int = 50) -> Dict[str, Any]:
        async with SessionLocal() as session:
            base_stmt = HistoryService._base_review_join_query()
            if teacher_id:
                base_stmt = base_stmt.where(DocumentRecord.owner_id == teacher_id)

            rows = (
                await session.execute(base_stmt.order_by(desc(ReviewRecord.created_at)).limit(limit))
            ).all()

            total_reviews, avg_score = await HistoryService._score_summary(session, owner_id=teacher_id)

            needs_attention = 0
            items = []
            for doc, review, revision in rows:
                if review.overall_score < 60:
                    needs_attention += 1
                items.append(HistoryService._teacher_dashboard_row(doc, review, revision))

            return {
                "teacher_id": teacher_id,
                "total_reviews": total_reviews,
                "average_score": avg_score,
                "needs_attention_count": needs_attention,
                "recent_reviews": items,
            }

    @staticmethod
    async def get_review_detail(review_id: UUID) -> Dict[str, Any] | None:
        async with SessionLocal() as session:
            stmt = (
                HistoryService._base_review_join_query()
                .where(ReviewRecord.id == review_id)
            )
            row = (await session.execute(stmt)).first()
            if not row:
                return None

            doc, review, revision = row
            return HistoryService._review_detail_payload(doc, review, revision)
