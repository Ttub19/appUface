"""ORM models for barber-shop discovery, booking, staff, and chat."""
import enum, uuid
from datetime import datetime, time
from decimal import Decimal
from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Index, Integer, Numeric, String, Text, Time, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base
from app.models import User
class ShopMemberRole(str,enum.Enum): owner="owner"; barber="barber"
class BookingStatus(str,enum.Enum): pending="pending"; confirmed="confirmed"; rejected="rejected"; cancelled="cancelled"; completed="completed"
def pk(): return mapped_column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)
class Shop(Base):
    __tablename__="barber_shops"
    id:Mapped[uuid.UUID]=pk(); owner_user_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id",ondelete="RESTRICT"),nullable=False,index=True)
    name:Mapped[str]=mapped_column(String(160),nullable=False,index=True); description:Mapped[str|None]=mapped_column(Text); address:Mapped[str]=mapped_column(String(500),nullable=False)
    latitude:Mapped[float]=mapped_column(Float,nullable=False); longitude:Mapped[float]=mapped_column(Float,nullable=False); phone:Mapped[str|None]=mapped_column(String(40))
    specialties:Mapped[list[str]]=mapped_column(JSONB,nullable=False,default=list,server_default=text("'[]'::jsonb")); timezone:Mapped[str]=mapped_column(String(64),nullable=False,default="Asia/Bangkok"); is_active:Mapped[bool]=mapped_column(Boolean,nullable=False,default=True)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now()); updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now())
    members:Mapped[list["ShopMember"]]=relationship(back_populates="shop",cascade="all, delete-orphan"); services:Mapped[list["ShopService"]]=relationship(back_populates="shop",cascade="all, delete-orphan"); hours:Mapped[list["BusinessHour"]]=relationship(back_populates="shop",cascade="all, delete-orphan")
    __table_args__=(CheckConstraint("latitude BETWEEN -90 AND 90",name="ck_shop_lat"),CheckConstraint("longitude BETWEEN -180 AND 180",name="ck_shop_lng"))
class ShopMember(Base):
    __tablename__="shop_members"
    id:Mapped[uuid.UUID]=pk(); shop_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("barber_shops.id",ondelete="CASCADE"),nullable=False,index=True); user_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id",ondelete="CASCADE"),nullable=False,index=True)
    role:Mapped[ShopMemberRole]=mapped_column(Enum(ShopMemberRole,name="shop_member_role"),nullable=False,default=ShopMemberRole.barber); bio:Mapped[str|None]=mapped_column(Text); specialties:Mapped[list[str]]=mapped_column(JSONB,nullable=False,default=list,server_default=text("'[]'::jsonb")); years_experience:Mapped[int]=mapped_column(Integer,nullable=False,default=0); is_active:Mapped[bool]=mapped_column(Boolean,nullable=False,default=True); created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
    shop:Mapped[Shop]=relationship(back_populates="members"); user:Mapped[User]=relationship(foreign_keys=[user_id])
    __table_args__=(UniqueConstraint("shop_id","user_id",name="uq_shop_member"),UniqueConstraint("user_id",name="uq_shop_member_one_shop"),CheckConstraint("years_experience BETWEEN 0 AND 80",name="ck_staff_years"))
class ShopService(Base):
    __tablename__="shop_services"
    id:Mapped[uuid.UUID]=pk(); shop_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("barber_shops.id",ondelete="CASCADE"),nullable=False,index=True); name:Mapped[str]=mapped_column(String(160),nullable=False); description:Mapped[str|None]=mapped_column(Text); duration_minutes:Mapped[int]=mapped_column(Integer,nullable=False); price:Mapped[Decimal]=mapped_column(Numeric(10,2),nullable=False); is_active:Mapped[bool]=mapped_column(Boolean,nullable=False,default=True); created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())
    shop:Mapped[Shop]=relationship(back_populates="services"); __table_args__=(UniqueConstraint("shop_id","name",name="uq_shop_service"),CheckConstraint("duration_minutes BETWEEN 15 AND 480",name="ck_service_duration"),CheckConstraint("price>=0",name="ck_service_price"))
class BusinessHour(Base):
    __tablename__="shop_business_hours"
    id:Mapped[uuid.UUID]=pk(); shop_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("barber_shops.id",ondelete="CASCADE"),nullable=False,index=True); day_of_week:Mapped[int]=mapped_column(Integer,nullable=False); open_time:Mapped[time|None]=mapped_column(Time); close_time:Mapped[time|None]=mapped_column(Time); is_closed:Mapped[bool]=mapped_column(Boolean,nullable=False,default=False)
    shop:Mapped[Shop]=relationship(back_populates="hours"); __table_args__=(UniqueConstraint("shop_id","day_of_week",name="uq_shop_day"),CheckConstraint("day_of_week BETWEEN 0 AND 6",name="ck_shop_day"),CheckConstraint("is_closed OR (open_time IS NOT NULL AND close_time IS NOT NULL AND close_time>open_time)",name="ck_shop_hours"))
class Booking(Base):
    __tablename__="barber_bookings"
    id:Mapped[uuid.UUID]=pk(); shop_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("barber_shops.id",ondelete="RESTRICT"),nullable=False,index=True); customer_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id",ondelete="RESTRICT"),nullable=False,index=True); barber_member_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("shop_members.id",ondelete="RESTRICT"),nullable=False,index=True); service_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("shop_services.id",ondelete="RESTRICT"),nullable=False)
    starts_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False,index=True); ends_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False); status:Mapped[BookingStatus]=mapped_column(Enum(BookingStatus,name="booking_status"),nullable=False,default=BookingStatus.pending,index=True); customer_note:Mapped[str|None]=mapped_column(Text); cancel_reason:Mapped[str|None]=mapped_column(Text); cancelled_by_id:Mapped[uuid.UUID|None]=mapped_column(ForeignKey("users.id",ondelete="SET NULL")); created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now()); updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now())
    __table_args__=(CheckConstraint("ends_at>starts_at",name="ck_booking_time"),Index("ix_booking_barber_time","barber_member_id","starts_at","ends_at"))
class ShopMessage(Base):
    __tablename__="shop_messages"
    id:Mapped[uuid.UUID]=pk(); shop_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("barber_shops.id",ondelete="CASCADE"),nullable=False,index=True); customer_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id",ondelete="CASCADE"),nullable=False,index=True); sender_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("users.id",ondelete="CASCADE"),nullable=False); body:Mapped[str]=mapped_column(String(1000),nullable=False); created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now(),index=True); read_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True)); __table_args__=(Index("ix_shop_message_conversation","shop_id","customer_id","created_at"),)
