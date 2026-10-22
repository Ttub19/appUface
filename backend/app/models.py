"""
ORM models.

Table names match the ones already referenced in the front-end's mock
messages (`face_analytics`) where applicable, so the schema reads as the
natural "real" backend behind the existing mockup.
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class UserRole(str, enum.Enum):
    """Only registered roles live in the DB. "guest" in the front-end just means "no token sent"."""
    member = "member"
    barber = "barber"
    admin = "admin"


class FaceShape(str, enum.Enum):
    oval = "oval"
    round = "round"
    square = "square"
    heart = "heart"
    oblong = "oblong"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, name="user_role"), default=UserRole.member, nullable=False)

    # Barber applicants remain members until an admin approves their form.
    barber_verification_status: Mapped[str] = mapped_column(String(16), default="none", server_default="none", nullable=False, index=True)
    barber_legal_name: Mapped[str | None] = mapped_column(String(160))
    barber_phone: Mapped[str | None] = mapped_column(String(40))
    barber_years_experience: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    barber_specialties: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"), nullable=False)
    barber_application_note: Mapped[str | None] = mapped_column(Text)
    barber_review_note: Mapped[str | None] = mapped_column(Text)
    barber_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    barber_reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    # Profile / theme settings (mirrors the "โปรไฟล์ & ธีมสี" screen)
    notify_on_complete: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    theme_accent: Mapped[str] = mapped_column(String(9), default="#3E5C9A", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("barber_verification_status IN ('none','pending','approved','rejected')", name="ck_user_barber_verification"),
        CheckConstraint("barber_years_experience BETWEEN 0 AND 80", name="ck_user_barber_experience"),
    )

    scans: Mapped[list["ScanHistory"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    favorites: Mapped[list["Favorite"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    golden_ratio_results: Mapped[list["GoldenRatioResult"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class ScanHistory(Base):
    """Corresponds to the front-end's `face_analytics` mock table / history screen."""
    __tablename__ = "face_analytics"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    face_shape: Mapped[FaceShape] = mapped_column(Enum(FaceShape, name="face_shape"), nullable=False)
    face_shape_name: Mapped[str] = mapped_column(String(120), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    gender_pref: Mapped[str] = mapped_column(String(10), nullable=True)

    # Demo-only: storing the captured photo as a base64 data URL directly in
    # Postgres. In a real deployment, upload to object storage (S3/GCS) and
    # store just the URL/key here instead.
    image_data_url: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    user: Mapped["User"] = relationship(back_populates="scans")


class Favorite(Base):
    __tablename__ = "favorites"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    face_shape: Mapped[FaceShape] = mapped_column(Enum(FaceShape, name="face_shape"), nullable=False)
    face_shape_name: Mapped[str] = mapped_column(String(120), nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)  # stored pre-formatted, e.g. "92.3%"
    image_data_url: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="favorites")


class GoldenRatioResult(Base):
    """Optional persistence for the Golden Ratio Analysis screen's live metrics."""
    __tablename__ = "golden_ratio_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    forehead_pct: Mapped[float] = mapped_column(Float, nullable=False)
    midface_pct: Mapped[float] = mapped_column(Float, nullable=False)
    lower_pct: Mapped[float] = mapped_column(Float, nullable=False)
    length_width_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    ratio_score: Mapped[float] = mapped_column(Float, nullable=False)
    symmetry_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="golden_ratio_results")


class Feedback(Base):
    """Suggestions, bug reports and "contact admin" messages sent from the app.

    Guests may send one too (user_id is NULL), so the contact name/email are
    stored on the row itself. The admin works through them with `status` and can
    leave a reply in `admin_note`, which the sender sees under "ข้อความของฉัน".
    """
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    contact_name: Mapped[str] = mapped_column(String(120), nullable=False)
    contact_email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    subject: Mapped[str] = mapped_column(String(160), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[str | None] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(16), default="new", server_default="new", nullable=False, index=True)
    admin_note: Mapped[str | None] = mapped_column(Text)
    handled_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    handled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        CheckConstraint("category IN ('suggestion','bug','contact','other')", name="ck_feedback_category"),
        CheckConstraint("status IN ('new','read','resolved')", name="ck_feedback_status"),
    )
