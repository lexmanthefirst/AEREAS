from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Academic Writing Evaluation API"
    APP_VERSION: str = "2.1.0"
    DEBUG: bool = False
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # S3/MinIO Configuration
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "documents"

    # Database
    DATABASE_URL: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379"

    # LLM Configuration
    LLM_PROVIDER: str = "openrouter"
    GEMINI_API_KEY: Optional[str] = None
    USE_MODELS: bool = True
    USE_LLM_SYNTHESIS: bool = True
    SYNTHESIS_MODEL_NAME: str = "openai/gpt-4.1-mini"
    REVIEW_MODEL_NAME: str = "openai/gpt-4.1-mini"
    REVISION_MODEL_NAME: str = "openai/gpt-4.1-mini"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_HTTP_REFERER: Optional[str] = None
    OPENROUTER_X_TITLE: str = "Academic Writing Evaluation API"

    # Worker LLM
    WORKER_MODEL_NAME: str = "openai/gpt-4.1-mini"
    LLM_REQUEST_TIMEOUT: float = 45.0

    # Research providers
    ENABLE_WEB_RESEARCH: bool = True
    WEB_RESEARCH_MAX_QUERIES: int = 3
    DUCKDUCKGO_API_URL: str = "https://api.duckduckgo.com/"
    CROSSREF_API_URL: str = "https://api.crossref.org/works"
    RESEARCH_USER_AGENT: str = "academic-review-system/0.1.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
