from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from sqlalchemy import DateTime, Select, asc, desc as sa_desc, select
from sqlalchemy import func as sa_func
from sqlalchemy.orm import DeclarativeBase, InstrumentedAttribute, Mapped, mapped_column, selectinload
from sqlalchemy.sql import func

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T", bound="BaseModel")


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class QueryBuilder(Generic[T]):
    """Fluent, chainable SELECT builder."""

    def __init__(self, model_class: type[T], db: AsyncSession) -> None:
        self.model_class = model_class
        self.db = db
        self._query: Select[Any] = select(model_class)

    def filter(self, *conditions: Any) -> QueryBuilder[T]:
        self._query = self._query.where(*conditions)
        return self

    def filter_by(self, **kwargs: Any) -> QueryBuilder[T]:
        self._query = self._query.filter_by(**kwargs)
        return self

    def with_relations(self, *relations: str | InstrumentedAttribute[Any]) -> QueryBuilder[T]:
        for relation in relations:
            if isinstance(relation, str):
                self._query = self._query.options(selectinload(getattr(self.model_class, relation)))
            else:
                self._query = self._query.options(selectinload(relation))
        return self

    def order_by(self, *columns: Any, desc: bool = False) -> QueryBuilder[T]:
        ordered = [sa_desc(c) if desc else asc(c) for c in columns]
        self._query = self._query.order_by(*ordered)
        return self

    def limit(self, n: int) -> QueryBuilder[T]:
        self._query = self._query.limit(n)
        return self

    def offset(self, n: int) -> QueryBuilder[T]:
        self._query = self._query.offset(n)
        return self

    def paginate(self, page: int = 1, per_page: int = 20) -> QueryBuilder[T]:
        self._query = self._query.limit(per_page).offset((page - 1) * per_page)
        return self

    def with_status(self, status: Any) -> QueryBuilder[T]:
        self._query = self._query.where(getattr(self.model_class, "status") == status)
        return self

    def created_after(self, dt: datetime) -> QueryBuilder[T]:
        self._query = self._query.where(getattr(self.model_class, "created_at") >= dt)
        return self

    def created_before(self, dt: datetime) -> QueryBuilder[T]:
        self._query = self._query.where(getattr(self.model_class, "created_at") <= dt)
        return self

    def for_user(self, user_id: Any) -> QueryBuilder[T]:
        self._query = self._query.where(getattr(self.model_class, "user_id") == user_id)
        return self

    async def all(self) -> Sequence[T]:
        result = await self.db.execute(self._query)
        return result.scalars().all()

    async def first(self) -> T | None:
        result = await self.db.execute(self._query)
        return result.scalars().first()

    async def one(self) -> T:
        result = await self.db.execute(self._query)
        return result.scalars().one()

    async def one_or_none(self) -> T | None:
        result = await self.db.execute(self._query)
        return result.scalars().one_or_none()

    async def count(self) -> int:
        count_q = select(sa_func.count()).select_from(self._query.subquery())
        result = await self.db.execute(count_q)
        return int(result.scalar() or 0)

    async def count_distinct(self, column: Any) -> int:
        count_q = select(sa_func.count(sa_func.distinct(column))).select_from(self._query.subquery())
        result = await self.db.execute(count_q)
        return int(result.scalar() or 0)


class BaseModel(Base):
    """Abstract base with timestamps and async CRUD/query helpers."""

    __abstract__ = True

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def add(self, db: AsyncSession) -> BaseModel:
        db.add(self)
        return self

    async def insert(self, db: AsyncSession, *, commit: bool = False, flush: bool = True) -> BaseModel:
        db.add(self)
        if commit:
            await db.commit()
            await db.refresh(self)
        elif flush:
            await db.flush()
        return self

    async def save(self, db: AsyncSession, *, commit: bool = False, flush: bool = True) -> BaseModel:
        self.updated_at = datetime.now(tz=timezone.utc)
        db.add(self)
        if commit:
            await db.commit()
            await db.refresh(self)
        elif flush:
            await db.flush()
        return self

    async def delete(self, db: AsyncSession, *, commit: bool = False) -> BaseModel:
        await db.delete(self)
        if commit:
            await db.commit()
        else:
            await db.flush()
        return self

    @classmethod
    def query(cls: type[T], db: AsyncSession) -> QueryBuilder[T]:
        return QueryBuilder(cls, db)

    @classmethod
    async def fetch_one(cls: type[T], db: AsyncSession, **kwargs: Any) -> T | None:
        result = await db.execute(select(cls).filter_by(**kwargs))
        return result.scalars().first()

    @classmethod
    async def fetch_unique(cls: type[T], db: AsyncSession, **kwargs: Any) -> T | None:
        result = await db.execute(select(cls).filter_by(**kwargs))
        return result.scalars().one_or_none()

    @classmethod
    async def fetch_all(cls: type[T], db: AsyncSession, **kwargs: Any) -> Sequence[T]:
        result = await db.execute(select(cls).filter_by(**kwargs))
        return result.scalars().all()

    @classmethod
    async def fetch_by_id(cls: type[T], db: AsyncSession, id: Any) -> T | None:
        return await db.get(cls, id)


__all__ = ["Base", "BaseModel", "QueryBuilder"]
