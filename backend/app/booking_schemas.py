import uuid
from datetime import datetime,time
from decimal import Decimal
from typing import Literal
from pydantic import AwareDatetime,BaseModel,ConfigDict,EmailStr,Field,field_validator,model_validator
def tags(values):
    out=[];seen=set()
    for raw in values:
        value=raw.strip();key=value.casefold()
        if value and key not in seen: seen.add(key);out.append(value)
    if len(out)>20: raise ValueError("ระบุความเชี่ยวชาญได้ไม่เกิน 20 รายการ")
    return out
class ShopCreate(BaseModel):
    name:str=Field(min_length=2,max_length=160);description:str|None=Field(default=None,max_length=3000);address:str=Field(min_length=3,max_length=500);latitude:float=Field(ge=-90,le=90);longitude:float=Field(ge=-180,le=180);phone:str|None=Field(default=None,max_length=40);specialties:list[str]=Field(default_factory=list);timezone:str="Asia/Bangkok"
    @field_validator("specialties")
    @classmethod
    def clean(cls,v):return tags(v)
class StaffCreate(BaseModel):
    email:EmailStr;bio:str|None=Field(default=None,max_length=2000);specialties:list[str]=Field(default_factory=list);years_experience:int=Field(default=0,ge=0,le=80)
    @field_validator("specialties")
    @classmethod
    def clean(cls,v):return tags(v)
class ServiceCreate(BaseModel): name:str=Field(min_length=2,max_length=160);description:str|None=Field(default=None,max_length=2000);duration_minutes:int=Field(ge=15,le=480);price:Decimal=Field(ge=0,max_digits=10,decimal_places=2)
class HourIn(BaseModel):
    day_of_week:int=Field(ge=0,le=6);open_time:time|None=None;close_time:time|None=None;is_closed:bool=False
    @model_validator(mode="after")
    def valid(self):
        if not self.is_closed and (self.open_time is None or self.close_time is None or self.close_time<=self.open_time): raise ValueError("เวลาทำการไม่ถูกต้อง")
        return self
class HoursSet(BaseModel):
    hours:list[HourIn]=Field(min_length=1,max_length=7)
    @model_validator(mode="after")
    def unique(self):
        if len({x.day_of_week for x in self.hours})!=len(self.hours):raise ValueError("วันทำการซ้ำ")
        return self
class BookingCreate(BaseModel): shop_id:uuid.UUID;service_id:uuid.UUID;barber_member_id:uuid.UUID;starts_at:AwareDatetime;customer_note:str|None=Field(default=None,max_length=1000)
class Action(BaseModel): reason:str|None=Field(default=None,max_length=1000)
class MessageCreate(BaseModel):
    body:str=Field(min_length=1,max_length=1000)
    @field_validator("body")
    @classmethod
    def clean(cls,v):
        v=v.strip()
        if not v:raise ValueError("ข้อความต้องไม่ว่าง")
        return v
class ShopItem(BaseModel): id:uuid.UUID;name:str;description:str|None;address:str;latitude:float;longitude:float;phone:str|None;specialties:list[str];timezone:str;distance_km:float|None=None;service_count:int=0;barber_count:int=0
class StaffOut(BaseModel): id:uuid.UUID;user_id:uuid.UUID;display_name:str;role:Literal["owner","barber"];verification_status:Literal["none","pending","approved","rejected"];bio:str|None;specialties:list[str];years_experience:int;is_active:bool
class ServiceOut(BaseModel):
    model_config=ConfigDict(from_attributes=True)
    id:uuid.UUID;name:str;description:str|None;duration_minutes:int;price:Decimal;is_active:bool
class HourOut(BaseModel):
    model_config=ConfigDict(from_attributes=True)
    day_of_week:int;open_time:time|None;close_time:time|None;is_closed:bool
class ShopDetail(ShopItem): services:list[ServiceOut];staff:list[StaffOut];business_hours:list[HourOut];staff_role:Literal["owner","barber","admin"]|None=None
class SlotOut(BaseModel):barber_member_id:uuid.UUID;barber_name:str;starts_at:datetime;ends_at:datetime
class BookingOut(BaseModel):id:uuid.UUID;shop_id:uuid.UUID;shop_name:str;customer_id:uuid.UUID;customer_name:str;barber_member_id:uuid.UUID;barber_name:str;service_id:uuid.UUID;service_name:str;price:Decimal;starts_at:datetime;ends_at:datetime;status:Literal["pending","confirmed","rejected","cancelled","completed"];customer_note:str|None;cancel_reason:str|None;can_customer_cancel:bool;created_at:datetime
class MessageOut(BaseModel):id:uuid.UUID;shop_id:uuid.UUID;customer_id:uuid.UUID;sender_id:uuid.UUID;sender_name:str;sender_is_staff:bool;body:str;created_at:datetime
class ConversationOut(BaseModel):customer_id:uuid.UUID;customer_name:str;last_message:str;last_message_at:datetime
