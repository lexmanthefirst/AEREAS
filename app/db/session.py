from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


def _normalize_database_url(database_url: str) -> str:
    """Ensure SQLAlchemy async driver format for PostgreSQL URLs."""
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


DATABASE_URL = _normalize_database_url(
    settings.DATABASE_URL or "postgresql+asyncpg://user:pass@localhost:5432/evaldb"
)

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an AsyncSession (SQLAlchemy 2.0 style)."""
    async with SessionLocal() as session:
        yield session


__all__ = ["engine", "SessionLocal", "get_db_session"]
