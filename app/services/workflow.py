import io
import time
import uuid
from typing import Any, Dict

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.review import (
    EvaluationRequestSchema,
    EvaluationResponseSchema,
    ReviewRevisionResponseSchema,
)
from app.services.document import DocumentExtractor
from app.services.history import HistoryService
from app.services.revision import RevisionService
from app.services.storage import get_storage_service
from app.supervisor.agent import SupervisorAgent
from app.utils.logger import logger


class ReviewWorkflowService:
    """Service layer orchestration for evaluation, review, revision, and upload workflows."""

    @staticmethod
    def _action_payload(actions: list[Any], limit: int = 10) -> list[dict[str, Any]]:
        return [
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
            for action in actions[:limit]
        ]

    @staticmethod
    def _runtime_info(context: Any, supervisor: SupervisorAgent) -> Dict[str, Any]:
        profile = supervisor.get_runtime_profile()
        worker_stats = {
            name: {
                "processing_time_ms": round(result.processing_time_ms, 2),
                "findings_count": len(result.findings),
                "flagged_items_count": len(result.flagged_items),
            }
            for name, result in context.worker_results.items()
        }
        return {
            **profile,
            "worker_stats": worker_stats,
            "document_sections": len(getattr(context, "document_sections", [])),
        }

    @staticmethod
    def _validate_file(file: UploadFile) -> None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        if not DocumentExtractor.is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. Supported: "
                    f"{', '.join(DocumentExtractor.SUPPORTED_EXTENSIONS)}"
                ),
            )

    @staticmethod
    async def _load_uploaded_document(file: UploadFile) -> Dict[str, Any]:
        storage = get_storage_service()
        filename = file.filename or "unknown"
        file_id = str(uuid.uuid4())
        extension = filename.split(".")[-1] if "." in filename else "txt"
        storage_key = f"uploads/{file_id}.{extension}"

        # Read once from request stream to avoid reuse/seek issues on closed temp files.
        raw_content = await file.read()
        if not raw_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        storage.upload_bytes(
            raw_content,
            storage_key,
            content_type=file.content_type or "application/octet-stream",
        )
        file_data = io.BytesIO(raw_content)

        extracted = DocumentExtractor.extract_document(file_data, filename)
        content = extracted["content"]
        stripped = content.strip()
        logger.info(
            "Extracted content for %s: %d chars",
            filename,
            len(stripped),
        )
        if len(stripped) < 10:
            logger.warning(
                "Upload rejected for %s: insufficient extracted text (%d chars)",
                filename,
                len(stripped),
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Document contains insufficient text after extraction "
                    f"({len(stripped)} chars)."
                ),
            )
        return extracted

    @staticmethod
    async def evaluate_text(
        request: EvaluationRequestSchema,
        supervisor: SupervisorAgent,
    ) -> EvaluationResponseSchema:
        start_time = time.perf_counter()
        structured = DocumentExtractor.parse_text_structure(request.content)
        context = await supervisor.run_evaluation(
            content=structured["content"],
            document_structure=structured,
            citation_style=request.citation_style,
        )
        processing_time = (time.perf_counter() - start_time) * 1000

        return EvaluationResponseSchema(
            document_id=str(context.document_id),
            overall_score=context.overall_score or 0.0,
            worker_scores=context.final_scores,
            actions=ReviewWorkflowService._action_payload(context.final_actions),
            critic_approved=context.critic_review.approved if context.critic_review else True,
            synthesis_reasoning=context.synthesis_reasoning,
            runtime_info=ReviewWorkflowService._runtime_info(context, supervisor),
            processing_time_ms=processing_time,
        )

    @staticmethod
    async def review_revise_text(
        request: EvaluationRequestSchema,
        supervisor: SupervisorAgent,
        session: AsyncSession,
    ) -> ReviewRevisionResponseSchema:
        start_time = time.perf_counter()
        structured = DocumentExtractor.parse_text_structure(request.content)
        context = await supervisor.run_evaluation(
            content=structured["content"],
            document_structure=structured,
            citation_style=request.citation_style,
        )
        revision_result = await RevisionService.revise_document(
            context, use_llm=request.use_llm_rewrite, llm_client=supervisor.llm_client,
        )
        processing_time = (time.perf_counter() - start_time) * 1000

        persisted = await HistoryService.persist_review_revision(
            session=session,
            context=context,
            revision_result=revision_result,
            processing_time_ms=processing_time,
            requester_id=request.requester_id,
            requester_role=request.requester_role,
            source_filename=None,
        )

        return ReviewRevisionResponseSchema(
            document_id=persisted["document_id"],
            overall_score=context.overall_score or 0.0,
            worker_scores=context.final_scores,
            actions=ReviewWorkflowService._action_payload(context.final_actions),
            critic_approved=context.critic_review.approved if context.critic_review else True,
            synthesis_reasoning=context.synthesis_reasoning,
            runtime_info=ReviewWorkflowService._runtime_info(context, supervisor),
            revised_content=revision_result["revised_content"],
            revision_mode=revision_result["revision_mode"],
            rewrite_summary=revision_result["rewrite_summary"],
            quality_metrics=revision_result["quality_metrics"],
            tracked_changes=revision_result["tracked_changes"],
            change_summary=revision_result["change_summary"],
            processing_time_ms=processing_time,
        )

    @staticmethod
    async def evaluate_upload(
        file: UploadFile,
        citation_style: str,
        supervisor: SupervisorAgent,
    ) -> EvaluationResponseSchema:
        ReviewWorkflowService._validate_file(file)
        start_time = time.perf_counter()
        extracted = await ReviewWorkflowService._load_uploaded_document(file)
        context = await supervisor.run_evaluation(
            content=extracted["content"],
            document_structure=extracted,
            citation_style=citation_style,
        )
        processing_time = (time.perf_counter() - start_time) * 1000

        return EvaluationResponseSchema(
            document_id=str(context.document_id),
            overall_score=context.overall_score or 0.0,
            worker_scores=context.final_scores,
            actions=ReviewWorkflowService._action_payload(context.final_actions),
            critic_approved=context.critic_review.approved if context.critic_review else True,
            synthesis_reasoning=context.synthesis_reasoning,
            runtime_info=ReviewWorkflowService._runtime_info(context, supervisor),
            processing_time_ms=processing_time,
        )

    @staticmethod
    async def review_revise_upload(
        file: UploadFile,
        citation_style: str,
        requester_id: str | None,
        requester_role: str,
        use_llm_rewrite: bool,
        supervisor: SupervisorAgent,
        session: AsyncSession,
    ) -> ReviewRevisionResponseSchema:
        ReviewWorkflowService._validate_file(file)
        start_time = time.perf_counter()
        extracted = await ReviewWorkflowService._load_uploaded_document(file)
        context = await supervisor.run_evaluation(
            content=extracted["content"],
            document_structure=extracted,
            citation_style=citation_style,
        )
        revision_result = await RevisionService.revise_document(
            context, use_llm=use_llm_rewrite, llm_client=supervisor.llm_client,
        )
        processing_time = (time.perf_counter() - start_time) * 1000

        persisted = await HistoryService.persist_review_revision(
            session=session,
            context=context,
            revision_result=revision_result,
            processing_time_ms=processing_time,
            requester_id=requester_id,
            requester_role=requester_role,
            source_filename=file.filename,
        )

        return ReviewRevisionResponseSchema(
            document_id=persisted["document_id"],
            overall_score=context.overall_score or 0.0,
            worker_scores=context.final_scores,
            actions=ReviewWorkflowService._action_payload(context.final_actions),
            critic_approved=context.critic_review.approved if context.critic_review else True,
            synthesis_reasoning=context.synthesis_reasoning,
            runtime_info=ReviewWorkflowService._runtime_info(context, supervisor),
            revised_content=revision_result["revised_content"],
            revision_mode=revision_result["revision_mode"],
            rewrite_summary=revision_result["rewrite_summary"],
            quality_metrics=revision_result["quality_metrics"],
            tracked_changes=revision_result["tracked_changes"],
            change_summary=revision_result["change_summary"],
            processing_time_ms=processing_time,
        )

    @staticmethod
    async def extract_upload(file: UploadFile) -> Dict[str, Any]:
        ReviewWorkflowService._validate_file(file)
        content = await file.read()
        file_data = io.BytesIO(content)
        extracted = DocumentExtractor.extract_document(file_data, file.filename or "")
        text = extracted["content"]

        return {
            "filename": file.filename,
            "content": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "sections": [section.model_dump() for section in extracted["sections"]],
            "document_metadata": extracted["metadata"],
        }
