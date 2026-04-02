from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.1.0"}


@router.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Academic Writing Evaluation API",
        "version": "2.1.0",
        "docs": "/docs",
        "endpoints": {
            "evaluate": "POST /api/v1/evaluate",
            "health": "GET /api/v1/health",
        }
    }
