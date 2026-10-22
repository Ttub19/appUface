import uuid
from datetime import datetime,timezone
from typing import Literal
from fastapi import APIRouter,Depends,HTTPException,Query
from sqlalchemy import func,select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.deps import require_role
from app.models import ScanHistory,User,UserRole
from app.schemas import AdminStatsOut,BarberApplicationOut,BarberReview,MonthlyUsageItem,ShapeDistributionItem
router=APIRouter(prefix="/admin",tags=["admin"])
@router.get("/stats",response_model=AdminStatsOut,dependencies=[Depends(require_role("admin"))])
async def get_stats(db:AsyncSession=Depends(get_db)):
 users=(await db.execute(select(func.count()).select_from(User))).scalar_one();scans=(await db.execute(select(func.count()).select_from(ScanHistory))).scalar_one();avg=(await db.execute(select(func.avg(ScanHistory.confidence)))).scalar_one()
 shapes=(await db.execute(select(ScanHistory.face_shape,func.count()).group_by(ScanHistory.face_shape).order_by(func.count().desc()))).all();m=func.to_char(ScanHistory.created_at,"YYYY-MM");months=(await db.execute(select(m.label("month"),func.count()).group_by(m).order_by(m.desc()).limit(6))).all()
 return AdminStatsOut(total_users=users,total_scans=scans,average_confidence=round(float(avg),1) if avg else 0,shape_distribution=[ShapeDistributionItem(face_shape=s.value,count=c) for s,c in shapes],monthly_usage=[MonthlyUsageItem(month=x,count=c) for x,c in reversed(months)])
@router.get("/barber-applications",response_model=list[BarberApplicationOut])
async def applications(verification_status:Literal["pending","approved","rejected"]|None=Query("pending",alias="status"),admin:User=Depends(require_role("admin")),db:AsyncSession=Depends(get_db)):
 q=select(User).where(User.barber_verification_status!="none")
 if verification_status:q=q.where(User.barber_verification_status==verification_status)
 return list((await db.execute(q.order_by(User.created_at))).scalars())
async def applicant(db,user_id):
 u=(await db.execute(select(User).where(User.id==user_id).with_for_update())).scalar_one_or_none()
 if not u or u.barber_verification_status=="none":raise HTTPException(404,"ไม่พบคำขอสมัครบัญชีช่าง")
 return u
@router.post("/barber-applications/{user_id}/approve",response_model=BarberApplicationOut)
async def approve(user_id:uuid.UUID,p:BarberReview,admin:User=Depends(require_role("admin")),db:AsyncSession=Depends(get_db)):
 u=await applicant(db,user_id);u.role=UserRole.barber;u.barber_verification_status="approved";u.barber_review_note=p.review_note or "อนุมัติโดยผู้ดูแลระบบ";u.barber_reviewed_at=datetime.now(timezone.utc);u.barber_reviewed_by_id=admin.id;await db.commit();await db.refresh(u);return u
@router.post("/barber-applications/{user_id}/reject",response_model=BarberApplicationOut)
async def reject(user_id:uuid.UUID,p:BarberReview,admin:User=Depends(require_role("admin")),db:AsyncSession=Depends(get_db)):
 u=await applicant(db,user_id)
 if u.barber_verification_status=="approved":raise HTTPException(409,"บัญชีที่อนุมัติแล้วไม่สามารถปฏิเสธจากหน้าคำขอ")
 note=(p.review_note or "").strip()
 if not note:raise HTTPException(422,"กรุณาระบุเหตุผลที่ไม่อนุมัติ")
 u.role=UserRole.member;u.barber_verification_status="rejected";u.barber_review_note=note;u.barber_reviewed_at=datetime.now(timezone.utc);u.barber_reviewed_by_id=admin.id;await db.commit();await db.refresh(u);return u
