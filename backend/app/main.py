from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app import booking_models
from app.config import get_settings
from app.database import Base,engine
from app.routers import admin,auth,booking,favorites,feedback,golden_ratio,profile,scans
from app.services.face_shape_predictor import get_predictor
import logging,threading
settings=get_settings()
def _warm_up_face_model():
 """Load TensorFlow + the face-shape model in the background so the first scan isn't slow; never blocks startup."""
 p=get_predictor()
 if not hasattr(p,"warm_up"):return
 try:p.warm_up()
 except Exception as exc:logging.getLogger(__name__).error("Face-shape model not ready: %s",exc)
@asynccontextmanager
async def lifespan(app):
 async with engine.begin() as c:
  await c.execute(text("DO $$ BEGIN IF EXISTS(SELECT 1 FROM pg_type WHERE typname='user_role') THEN ALTER TYPE user_role ADD VALUE IF NOT EXISTS 'barber'; END IF; END $$;"))
 ddl=["ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_verification_status VARCHAR(16) NOT NULL DEFAULT 'none'","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_legal_name VARCHAR(160)","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_phone VARCHAR(40)","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_years_experience INTEGER NOT NULL DEFAULT 0","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_specialties JSONB NOT NULL DEFAULT '[]'::jsonb","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_application_note TEXT","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_review_note TEXT","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_reviewed_at TIMESTAMPTZ","ALTER TABLE IF EXISTS users ADD COLUMN IF NOT EXISTS barber_reviewed_by_id UUID"]
 async with engine.begin() as c:
  for s in ddl:await c.execute(text(s))
 async with engine.begin() as c:await c.run_sync(Base.metadata.create_all)
 async with engine.begin() as c:
  await c.execute(text("UPDATE users u SET role='barber'::user_role,barber_verification_status='approved',barber_legal_name=COALESCE(u.barber_legal_name,u.display_name),barber_review_note=COALESCE(u.barber_review_note,'แปลงสิทธิ์จากเจ้าของร้านเดิมอัตโนมัติ'),barber_reviewed_at=COALESCE(u.barber_reviewed_at,now()) WHERE EXISTS(SELECT 1 FROM barber_shops s WHERE s.owner_user_id=u.id)"))
  await c.execute(text("UPDATE users u SET barber_verification_status='pending',barber_legal_name=COALESCE(u.barber_legal_name,u.display_name),barber_application_note=COALESCE(u.barber_application_note,'บัญชีช่างเดิม กรุณาตรวจสอบก่อนเปิดสิทธิ์') WHERE u.role='member'::user_role AND u.barber_verification_status='none' AND EXISTS(SELECT 1 FROM shop_members sm WHERE sm.user_id=u.id AND sm.role='barber'::shop_member_role)"))
 threading.Thread(target=_warm_up_face_model,name="face-model-warmup",daemon=True).start()
 yield
app=FastAPI(title=settings.app_name,description="Face analysis and verified barber booking API",version="1.2.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origin_list,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
for r in (auth,scans,favorites,golden_ratio,profile,admin,booking,feedback):app.include_router(r.router)
@app.get("/health",tags=["health"])
async def health_check():return {"status":"ok"}
