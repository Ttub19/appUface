from fastapi import APIRouter,Depends,HTTPException,status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import get_settings
from app.database import get_db
from app.deps import get_current_user
from app.models import User,UserRole
from app.schemas import Token,UserCreate,UserOut
from app.security import create_access_token,hash_password,verify_password
router=APIRouter(prefix="/auth",tags=["auth"])
@router.post("/register",response_model=UserOut,status_code=status.HTTP_201_CREATED)
async def register(p:UserCreate,db:AsyncSession=Depends(get_db)):
 if (await db.execute(select(User).where(User.email==p.email))).scalar_one_or_none():raise HTTPException(409,"อีเมลนี้ถูกใช้งานแล้ว")
 barber=p.account_type=="barber"
 u=User(email=p.email,hashed_password=hash_password(p.password),display_name=p.display_name,role=UserRole.member,barber_verification_status="pending" if barber else "none",barber_legal_name=p.barber_legal_name if barber else None,barber_phone=p.barber_phone if barber else None,barber_years_experience=p.barber_years_experience if barber else 0,barber_specialties=p.barber_specialties if barber else [],barber_application_note=p.barber_application_note if barber else None)
 db.add(u);await db.commit();await db.refresh(u);return u
@router.post("/login",response_model=Token)
async def login(form_data:OAuth2PasswordRequestForm=Depends(),db:AsyncSession=Depends(get_db)):
 u=(await db.execute(select(User).where(User.email==form_data.username))).scalar_one_or_none()
 if not u or not verify_password(form_data.password,u.hashed_password):raise HTTPException(401,"อีเมลหรือรหัสผ่านไม่ถูกต้อง",headers={"WWW-Authenticate":"Bearer"})
 if u.email.lower() in get_settings().admin_email_list and u.role!=UserRole.admin:
  u.role=UserRole.admin;await db.commit();await db.refresh(u)
 return Token(access_token=create_access_token(str(u.id),{"role":u.role.value}))
@router.get("/me",response_model=UserOut)
async def me(current_user:User=Depends(get_current_user)):return current_user
