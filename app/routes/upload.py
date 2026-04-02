from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form

from app.schemas.review import EvaluationResponseSchema, ReviewRevisionResponseSchema
from app.services.workflow import ReviewWorkflowService
from app.routes.evaluation import get_supervisor
from app.supervisor.agent import SupervisorAgent
from app.utils.logger import logger


router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=EvaluationResponseSchema)
async def upload_and_evaluate(
    file: UploadFile = File(..., description="Document file (.txt, .docx, .pdf)"),
    citation_style: str = Form(default="harvard", description="Citation style"),
    supervisor: SupervisorAgent = Depends(get_supervisor),
):
    """
    Upload a document file and evaluate it.
    
    Supported formats:
    - .txt (plain text)
    - .md (markdown)
    - .docx (Microsoft Word)
    - .pdf (PDF document)
    
    The file is stored in MinIO/S3 and text is extracted for evaluation.
    """
    try:
        return await ReviewWorkflowService.evaluate_upload(
            file=file,
            citation_style=citation_style,
            supervisor=supervisor,
        )

    except HTTPException as e:
        logger.warning("Upload failed for %s: %s", file.filename, e.detail)
        raise
    
    except ValueError as e:
        logger.warning("Upload parsing failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        logger.error("Upload dependency error for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")
    except Exception as e:
        logger.exception("Upload evaluation failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/upload/review-revise", response_model=ReviewRevisionResponseSchema)
async def upload_review_and_revise(
    file: UploadFile = File(..., description="Document file (.txt, .docx, .pdf)"),
    citation_style: str = Form(default="harvard", description="Citation style"),
    requester_id: str | None = Form(default=None, description="Student or teacher identifier"),
    requester_role: str = Form(default="student", description="Role: student or teacher"),
    use_llm_rewrite: bool = Form(
        default=True,
        description="Enable LLM rewrite for critical/moderate issues",
    ),
    supervisor: SupervisorAgent = Depends(get_supervisor),
):
    """Upload a document, return feedback, and produce a revised draft."""
    try:
        return await ReviewWorkflowService.review_revise_upload(
            file=file,
            citation_style=citation_style,
            requester_id=requester_id,
            requester_role=requester_role,
            use_llm_rewrite=use_llm_rewrite,
            supervisor=supervisor,
        )

    except HTTPException as e:
        logger.warning("Review-revise upload failed for %s: %s", file.filename, e.detail)
        raise

    except ValueError as e:
        logger.warning("Review-revise parsing failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        logger.error("Review-revise dependency error for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")
    except Exception as e:
        logger.exception("Review-revise upload failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Review and revision failed: {str(e)}")


@router.post("/upload/extract")
async def upload_and_extract_only(
    file: UploadFile = File(..., description="Document file to extract text from"),
):
    """
    Upload a document and return extracted text only (no evaluation).
    Useful for previewing what will be evaluated.
    """
    try:
        return await ReviewWorkflowService.extract_upload(file=file)

    except HTTPException as e:
        logger.warning("Extract-only upload failed for %s: %s", file.filename, e.detail)
        raise
    
    except ValueError as e:
        logger.warning("Extract-only parsing failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        logger.error("Extract-only dependency error for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")
