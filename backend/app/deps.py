"""
FastAPI dependencies for authentication and role-based access control.
"""
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.security import decode_access_token

# tokenUrl is just the docs hint for Swagger UI's "Authorize" button.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


async def _load_user(db: AsyncSession, user_id: str) -> Optional[User]:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return None
    result = await db.execute(select(User).where(User.id == uid))
    return result.scalar_one_or_none()


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Requires a valid bearer token. Raises 401 otherwise."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="ไม่ได้เข้าสู่ระบบหรือ token ไม่ถูกต้อง",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_error

    claims = decode_access_token(token)
    if claims is None or "sub" not in claims:
        raise credentials_error

    user = await _load_user(db, claims["sub"])
    if user is None:
        raise credentials_error
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Like get_current_user, but returns None instead of raising when no/invalid
    token is present — used by endpoints the front-end's "guest" role can hit
    (e.g. running a prediction), which only need a real user to *save* results.
    """
    if token is None:
        return None
    claims = decode_access_token(token)
    if claims is None or "sub" not in claims:
        return None
    return await _load_user(db, claims["sub"])


def require_role(*roles: str):
    """Dependency factory: raises 403 unless the current user's role is in `roles`."""

    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role.value not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="คุณไม่มีสิทธิ์เข้าถึงข้อมูลนี้",
            )
        return current_user

    return _check
