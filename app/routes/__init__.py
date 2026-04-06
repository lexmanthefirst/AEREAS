from fastapi import APIRouter

from app.routes.evaluation import router as evaluation_router
from app.routes.health import router as health_router
from app.routes.upload import router as upload_router
from app.routes.dashboard import router as dashboard_router


# Create main router to include all sub-routers
api_router = APIRouter()

# Include all routers
api_router.include_router(health_router)
api_router.include_router(evaluation_router)
api_router.include_router(upload_router)
api_router.include_router(dashboard_router)


__all__ = [
    "api_router",
]
