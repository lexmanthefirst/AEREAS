from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import api_router, set_supervisor
from app.supervisor.agent import SupervisorAgent
from app.utils.logger import logger
from app.middleware.correlation import CorrelationIdMiddleware
from app.db import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup: Initialize supervisor
    logger.info("Initializing SupervisorAgent...")
    supervisor = SupervisorAgent(
        use_models=True,
        use_llm_synthesis=True,
    )
    set_supervisor(supervisor)
    logger.info("SupervisorAgent ready!")

    yield

    # Shutdown: Cleanup
    await engine.dispose()
    logger.info("Shutting down...")


app = FastAPI(
    title="Academic Writing Evaluation and Review API",
    description="Multi-agent system for comprehensive academic writing evaluation  and review system.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Correlation middleware for request tracing across logs
app.add_middleware(CorrelationIdMiddleware)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes from the routes package
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
