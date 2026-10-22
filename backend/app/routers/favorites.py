"""
POST   /favorites          — add a favorite
GET    /favorites          — list the current user's favorites
DELETE /favorites/{id}     — remove a favorite
"""
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import Favorite, User
from app.schemas import FavoriteCreate, FavoriteOut

router = APIRouter(prefix="/favorites", tags=["favorites"])


@router.post("", response_model=FavoriteOut, status_code=status.HTTP_201_CREATED)
async def create_favorite(
    payload: FavoriteCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Favorite:
    favorite = Favorite(user_id=current_user.id, **payload.model_dump())
    db.add(favorite)
    await db.commit()
    await db.refresh(favorite)
    return favorite


@router.get("", response_model=List[FavoriteOut])
async def list_favorites(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[Favorite]:
    result = await db.execute(
        select(Favorite).where(Favorite.user_id == current_user.id).order_by(Favorite.created_at.desc())
    )
    return list(result.scalars().all())


@router.delete("/{favorite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_favorite(
    favorite_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(Favorite).where(Favorite.id == favorite_id, Favorite.user_id == current_user.id)
    )
    favorite = result.scalar_one_or_none()
    if favorite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ไม่พบรายการโปรดนี้")

    await db.delete(favorite)
    await db.commit()
