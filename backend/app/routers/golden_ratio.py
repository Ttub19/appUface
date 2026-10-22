"""
POST /golden-ratio — save a Golden Ratio analysis snapshot computed client-side
GET  /golden-ratio — list the current user's past analyses
"""
from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import GoldenRatioResult, User
from app.schemas import GoldenRatioCreate, GoldenRatioOut

router = APIRouter(prefix="/golden-ratio", tags=["golden-ratio"])


@router.post("", response_model=GoldenRatioOut, status_code=status.HTTP_201_CREATED)
async def save_golden_ratio_result(
    payload: GoldenRatioCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> GoldenRatioResult:
    """
    The Golden Ratio metrics themselves are computed client-side in real time
    from MediaPipe landmarks (see js/goldenRatioCalc.js in the front-end) —
    this endpoint just persists a snapshot the user chose to keep.
    """
    entry = GoldenRatioResult(user_id=current_user.id, **payload.model_dump())
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


@router.get("", response_model=List[GoldenRatioOut])
async def list_golden_ratio_results(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[GoldenRatioResult]:
    result = await db.execute(
        select(GoldenRatioResult)
        .where(GoldenRatioResult.user_id == current_user.id)
        .order_by(GoldenRatioResult.created_at.desc())
    )
    return list(result.scalars().all())
