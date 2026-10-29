"""
GET  /hair/realistic/config  — is "สร้างภาพสมจริง" available, which mode, how many left today
POST /hair/realistic         — redraw the hair on the user's photo with a generative model

Members only, HAIR_AI_DAILY_LIMIT per Bangkok day (only successful generations
count). The photo is sent to the configured outside service — the front-end
says so before the first use. See app/services/hair_ai.py.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models import HairGeneration, User
from app.schemas import HairAIConfigOut, HairAIRequest, HairAIResponse
from app.services import hair_ai

router = APIRouter(prefix="/hair", tags=["hair"])


async def _used_today(db: AsyncSession, user: User) -> int:
    q = select(func.count()).select_from(HairGeneration).where(
        HairGeneration.user_id == user.id, HairGeneration.created_at >= hair_ai.day_start_utc())
    return (await db.execute(q)).scalar_one()


@router.get("/realistic/config", response_model=HairAIConfigOut)
async def realistic_config(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> HairAIConfigOut:
    cfg = hair_ai.load_config()
    used = await _used_today(db, user) if cfg.enabled else 0
    return HairAIConfigOut(enabled=cfg.enabled, provider=cfg.provider, provider_label=hair_ai.PROVIDER_LABELS.get(cfg.provider),
                           mode=cfg.mode, daily_limit=cfg.daily_limit, used_today=used,
                           remaining_today=max(0, cfg.daily_limit - used))


@router.post("/realistic", response_model=HairAIResponse)
async def realistic_hair(payload: HairAIRequest, user: User = Depends(get_current_user),
                         db: AsyncSession = Depends(get_db)) -> HairAIResponse:
    cfg = hair_ai.load_config()
    if not cfg.enabled:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "ระบบสร้างภาพสมจริงยังไม่เปิดใช้งาน")
    used = await _used_today(db, user)
    if used >= cfg.daily_limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            f"วันนี้ใช้ครบ {cfg.daily_limit} รูปแล้ว พรุ่งนี้ใช้ได้ใหม่ (รีเซ็ตเที่ยงคืน)")

    color_words = hair_ai.describe_color(payload.color_id, payload.color_rgb)
    try:
        if cfg.mode == "patch":
            if not (payload.guide_data_url and payload.mask_data_url):
                raise ValueError("ต้องส่ง guide_data_url และ mask_data_url")
            guide, _ = hair_ai.decode_image_data_url(payload.guide_data_url, "guide")
            mask, _ = hair_ai.decode_image_data_url(payload.mask_data_url, "mask")
            for name, raw in (("guide", guide), ("mask", mask)):
                if hair_ai.image_size(raw) != (hair_ai.PATCH_SIZE, hair_ai.PATCH_SIZE):
                    raise ValueError(f"{name}: ต้องมีขนาด {hair_ai.PATCH_SIZE}x{hair_ai.PATCH_SIZE}")
            kwargs = {"guide": guide, "mask": mask}
        else:
            if not (payload.photo_data_url and payload.reference_data_url):
                raise ValueError("ต้องส่ง photo_data_url และ reference_data_url")
            kwargs = {"photo": hair_ai.decode_image_data_url(payload.photo_data_url, "photo"),
                      "reference": hair_ai.decode_image_data_url(payload.reference_data_url, "reference")}
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    try:
        raw, mime = await run_in_threadpool(hair_ai.call_provider, cfg, style_name=payload.style_name,
                                            color_words=color_words, **kwargs)
    except hair_ai.HairAIError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    db.add(HairGeneration(user_id=user.id, provider=cfg.provider, style_id=payload.style_id))
    await db.commit()
    return HairAIResponse(image_data_url=hair_ai.to_data_url(raw, mime), provider=cfg.provider, mode=cfg.mode,
                          remaining_today=max(0, cfg.daily_limit - used - 1))
