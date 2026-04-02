import uuid

from sqlalchemy import Boolean, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import BaseModel


class DocumentRecord(BaseModel):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    owner_role: Mapped[str] = mapped_column(String(32), default="student", index=True)
    source_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    original_content: Mapped[str] = mapped_column(Text, nullable=False)

    reviews: Mapped[list["ReviewRecord"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class ReviewRecord(BaseModel):
    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    worker_scores: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    actions: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    critic_approved: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    document: Mapped[DocumentRecord] = relationship(back_populates="reviews")
    revision: Mapped["RevisionRecord"] = relationship(back_populates="review", cascade="all, delete-orphan", uselist=False)


class RevisionRecord(BaseModel):
    __tablename__ = "revisions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    revised_content: Mapped[str] = mapped_column(Text, nullable=False)
    revision_mode: Mapped[str] = mapped_column(String(32), default="rules", nullable=False)
    rewrite_summary: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    quality_metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    tracked_changes: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    change_summary: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    review: Mapped[ReviewRecord] = relationship(back_populates="revision")


__all__ = ["DocumentRecord", "ReviewRecord", "RevisionRecord"]
