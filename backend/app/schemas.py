"""
Pydantic schemas — request/response contracts for the API.
Kept separate from the ORM models (app/models.py) on purpose, so the DB
schema and the wire format can evolve independently.
"""
import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

FaceShapeId = Literal["oval", "round", "square", "heart", "oblong"]


# ---------------------------------------------------------------------------
# Auth / users
# ---------------------------------------------------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=120)
    account_type: Literal["member", "barber"] = "member"
    barber_legal_name: Optional[str] = Field(default=None, min_length=2, max_length=160)
    barber_phone: Optional[str] = Field(default=None, min_length=8, max_length=40)
    barber_years_experience: int = Field(default=0, ge=0, le=80)
    barber_specialties: list[str] = Field(default_factory=list)
    barber_application_note: Optional[str] = Field(default=None, min_length=10, max_length=2000)

    @field_validator("barber_specialties")
    @classmethod
    def clean_tags(cls, values):
        out=[]; seen=set()
        for raw in values:
            value=raw.strip(); key=value.casefold()
            if value and key not in seen: seen.add(key); out.append(value)
        if len(out)>20: raise ValueError("ระบุความเชี่ยวชาญได้ไม่เกิน 20 รายการ")
        return out

    @model_validator(mode="after")
    def validate_barber_form(self):
        if self.account_type == "barber":
            if not self.barber_legal_name or not self.barber_phone:
                raise ValueError("กรุณากรอกชื่อจริงและเบอร์โทรสำหรับบัญชีช่าง")
            if not self.barber_specialties:
                raise ValueError("กรุณาระบุความเชี่ยวชาญอย่างน้อย 1 รายการ")
            if not self.barber_application_note:
                raise ValueError("กรุณากรอกข้อมูลแนะนำตัวสำหรับการตรวจสอบ")
        return self


BarberStatus = Literal["none", "pending", "approved", "rejected"]
class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: EmailStr
    display_name: str
    role: Literal["member", "barber", "admin"]
    barber_verification_status: BarberStatus
    barber_legal_name: Optional[str]
    barber_phone: Optional[str]
    barber_years_experience: int
    barber_specialties: list[str]
    barber_application_note: Optional[str]
    barber_review_note: Optional[str]
    notify_on_complete: bool
    theme_accent: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class BarberApplicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: EmailStr
    display_name: str
    role: Literal["member", "barber", "admin"]
    barber_verification_status: BarberStatus
    barber_legal_name: Optional[str]
    barber_phone: Optional[str]
    barber_years_experience: int
    barber_specialties: list[str]
    barber_application_note: Optional[str]
    barber_review_note: Optional[str]
    created_at: datetime


class BarberReview(BaseModel):
    review_note: Optional[str] = Field(default=None, max_length=1000)


# ---------------------------------------------------------------------------
# Face-shape scan# ---------------------------------------------------------------------------
# Face-shape scan / prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Body for POST /scans/predict — accepts the same base64 data URL the front-end already produces."""
    image_data_url: str = Field(min_length=10, description="base64 data URL of the captured/uploaded photo")
    gender: Optional[Literal["male", "female"]] = Field(
        default=None, description="picks the gender-specific model when the server has one")


class PredictResponse(BaseModel):
    face_shape: FaceShapeId
    face_shape_name: str
    confidence: float
    model: str = "default"


class ScanCreate(BaseModel):
    face_shape: FaceShapeId
    face_shape_name: str
    confidence: float = Field(ge=0, le=100)
    gender_pref: Optional[Literal["male", "female"]] = None
    image_data_url: Optional[str] = None


class ScanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    face_shape: FaceShapeId
    face_shape_name: str
    confidence: float
    gender_pref: Optional[str]
    image_data_url: Optional[str]
    created_at: datetime


# ---------------------------------------------------------------------------
# Favorites
# ---------------------------------------------------------------------------
class FavoriteCreate(BaseModel):
    face_shape: FaceShapeId
    face_shape_name: str
    confidence: str
    image_data_url: Optional[str] = None


class FavoriteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    face_shape: FaceShapeId
    face_shape_name: str
    confidence: str
    image_data_url: Optional[str]
    created_at: datetime


# ---------------------------------------------------------------------------
# Golden Ratio results
# ---------------------------------------------------------------------------
class GoldenRatioCreate(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    forehead_pct: float = Field(ge=0, le=100)
    midface_pct: float = Field(ge=0, le=100)
    lower_pct: float = Field(ge=0, le=100)
    length_width_ratio: float = Field(gt=0, le=5)
    ratio_score: float = Field(ge=0, le=100)
    symmetry_score: float = Field(ge=0, le=100)
    overall_score: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_derived_scores(self) -> "GoldenRatioCreate":
        thirds = [self.forehead_pct, self.midface_pct, self.lower_pct]
        if abs(sum(thirds) - 100) > 0.75:
            raise ValueError("vertical face percentages must sum to 100")
        phi = (1 + 5 ** 0.5) / 2
        expected_ratio = max(0.0, min(100.0, 100 - abs(self.length_width_ratio - phi) * 60))
        if abs(self.ratio_score - expected_ratio) > 1.5:
            raise ValueError("ratio_score is inconsistent with length_width_ratio")
        third_score = max(0.0, min(100.0, 100 - sum(abs(v - 100 / 3) for v in thirds) / 3 * 6))
        expected_overall = third_score * 0.4 + self.ratio_score * 0.35 + self.symmetry_score * 0.25
        if abs(self.overall_score - expected_overall) > 1.5:
            raise ValueError("overall_score is inconsistent with component scores")
        return self


class GoldenRatioOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    forehead_pct: float
    midface_pct: float
    lower_pct: float
    length_width_ratio: float
    ratio_score: float
    symmetry_score: float
    overall_score: float
    created_at: datetime


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
class ProfileUpdate(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    notify_on_complete: Optional[bool] = None
    theme_accent: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")


# ---------------------------------------------------------------------------
# Admin stats
# ---------------------------------------------------------------------------
class ShapeDistributionItem(BaseModel):
    face_shape: FaceShapeId
    count: int


class MonthlyUsageItem(BaseModel):
    month: str  # "YYYY-MM"
    count: int


class AdminStatsOut(BaseModel):
    total_users: int
    total_scans: int
    average_confidence: float
    shape_distribution: List[ShapeDistributionItem]
    monthly_usage: List[MonthlyUsageItem]


# ---------------------------------------------------------------------------
# Feedback / contact admin
# ---------------------------------------------------------------------------
FeedbackCategory = Literal["suggestion", "bug", "contact", "other"]
FeedbackStatus = Literal["new", "read", "resolved"]


class FeedbackCreate(BaseModel):
    category: FeedbackCategory = "suggestion"
    subject: str = Field(min_length=3, max_length=160)
    message: str = Field(min_length=10, max_length=2000)
    # required only for guests; signed-in users are taken from their account
    contact_name: Optional[str] = Field(default=None, max_length=120)
    contact_email: Optional[EmailStr] = None
    page: Optional[str] = Field(default=None, max_length=60)

    @field_validator("subject", "message")
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("ห้ามเว้นว่าง")
        return v


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    category: FeedbackCategory
    subject: str
    message: str
    status: FeedbackStatus
    admin_note: Optional[str]
    handled_at: Optional[datetime]
    created_at: datetime


class FeedbackAdminOut(FeedbackOut):
    user_id: Optional[uuid.UUID]
    contact_name: str
    contact_email: str
    page: Optional[str]


class FeedbackUpdate(BaseModel):
    status: FeedbackStatus
    admin_note: Optional[str] = Field(default=None, max_length=2000)
