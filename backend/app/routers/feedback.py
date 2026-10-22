"""
POST  /feedback               — send a suggestion / bug report / message to the admin (guests too)
GET   /feedback/mine          — the signed-in user's own messages and the admin's replies
GET   /admin/feedback         — admin inbox (filter by ?status=new|read|resolved)
PATCH /admin/feedback/{id}    — admin marks read / resolved and leaves a reply
"""
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user, get_current_user_optional, require_role
from app.models import Feedback, User
from app.schemas import FeedbackAdminOut, FeedbackCreate, FeedbackOut, FeedbackUpdate
from app.services.feedback_rules import RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW, handled_fields, resolve_contact

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
async def send_feedback(
    payload: FeedbackCreate,
    user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
) -> Feedback:
    try:
        name, email = resolve_contact(user, payload.contact_name, payload.contact_email)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    since = datetime.now(timezone.utc) - RATE_LIMIT_WINDOW
    sender = or_(Feedback.user_id == user.id, Feedback.contact_email == email) if user else Feedback.contact_email == email
    recent = (await db.execute(select(func.count()).select_from(Feedback).where(sender, Feedback.created_at >= since))).scalar_one()
    if recent >= RATE_LIMIT_COUNT:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "ส่งข้อความถี่เกินไป กรุณารอสักครู่แล้วลองใหม่")

    item = Feedback(
        user_id=user.id if user else None, contact_name=name, contact_email=email,
        category=payload.category, subject=payload.subject, message=payload.message, page=payload.page,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.get("/feedback/mine", response_model=list[FeedbackOut])
async def my_feedback(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> list[Feedback]:
    rows = await db.execute(select(Feedback).where(Feedback.user_id == user.id).order_by(Feedback.created_at.desc()).limit(50))
    return list(rows.scalars())


@router.get("/admin/feedback", response_model=list[FeedbackAdminOut])
async def feedback_inbox(
    feedback_status: Optional[Literal["new", "read", "resolved"]] = Query(None, alias="status"),
    admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
) -> list[Feedback]:
    q = select(Feedback)
    if feedback_status:
        q = q.where(Feedback.status == feedback_status)
    rows = await db.execute(q.order_by(Feedback.created_at.desc()).limit(200))
    return list(rows.scalars())


@router.patch("/admin/feedback/{feedback_id}", response_model=FeedbackAdminOut)
async def update_feedback(
    feedback_id: uuid.UUID,
    payload: FeedbackUpdate,
    admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
) -> Feedback:
    item = (await db.execute(select(Feedback).where(Feedback.id == feedback_id))).scalar_one_or_none()
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "ไม่พบข้อความนี้")
    item.status = payload.status
    if payload.admin_note is not None:
        item.admin_note = payload.admin_note.strip() or None
    item.handled_by_id, item.handled_at = handled_fields(payload.status, admin.id, datetime.now(timezone.utc))
    await db.commit()
    await db.refresh(item)
    return item
