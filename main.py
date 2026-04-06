from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import api_router
from app.llm import LLMClient
from app.supervisor.agent import SupervisorAgent
from app.utils.logger import logger
from app.middleware.correlation import CorrelationIdMiddleware
from app.db import engine
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup: Initialize supervisor
    logger.info("Initializing LLM client...")
    llm_client = LLMClient()
    app.state.llm_client = llm_client

    logger.info("Initializing SupervisorAgent...")
    supervisor = SupervisorAgent(
        llm_client=llm_client,
        use_llm_synthesis=settings.USE_LLM_SYNTHESIS,
    )
    app.state.supervisor = supervisor
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
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
