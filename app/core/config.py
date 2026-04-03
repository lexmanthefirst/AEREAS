from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "Academic Writing Evaluation API"
    APP_VERSION: str = "2.1.0"
    DEBUG: bool = False

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
    GEMINI_API_KEY: Optional[str] = None
    USE_MODELS: bool = True
    USE_LLM_SYNTHESIS: bool = True
    SYNTHESIS_MODEL_NAME: str = "gemini-2.5-flash"
    REVIEW_MODEL_NAME: str = "gemini-2.5-flash"
    REVISION_MODEL_NAME: str = "gemini-2.5-flash"

    # NLP model identifiers
    GRAMMAR_MODEL_NAME: str = "vennify/t5-base-grammar-correction"
    ARGUMENTATION_MODEL_NAME: str = "microsoft/deberta-v3-base"
    TONE_MODEL_NAME: str = "bert-base-uncased"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Research providers
    ENABLE_WEB_RESEARCH: bool = True
    WEB_RESEARCH_MAX_QUERIES: int = 3
    DUCKDUCKGO_API_URL: str = "https://api.duckduckgo.com/"
    CROSSREF_API_URL: str = "https://api.crossref.org/works"
    RESEARCH_USER_AGENT: str = "academic-review-system/0.1.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
