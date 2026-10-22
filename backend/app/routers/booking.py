import asyncio,uuid
from collections import defaultdict
from types import SimpleNamespace
from datetime import date,datetime,time,timedelta,timezone
from zoneinfo import ZoneInfo,ZoneInfoNotFoundError
from fastapi import APIRouter,Depends,HTTPException,Query,WebSocket,WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from sqlalchemy import func,or_,select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased
from app.booking_models import Booking,BookingStatus,BusinessHour,Shop,ShopMember,ShopMemberRole,ShopMessage,ShopService
from app.booking_schemas import Action,BookingCreate,BookingOut,ConversationOut,HourOut,HoursSet,MessageCreate,MessageOut,ServiceCreate,ServiceOut,ShopCreate,ShopDetail,ShopItem,SlotOut,StaffCreate,StaffOut
from app.database import AsyncSessionLocal,get_db
from app.deps import get_current_user,get_current_user_optional
from app.models import User,UserRole
from app.security import decode_access_token
from app.services.booking_rules import can_customer_cancel,haversine_km,slots
from app.services.barber_roles import approved,can_create_shop
router=APIRouter(tags=["barber booking"]);ACTIVE=(BookingStatus.pending,BookingStatus.confirmed)
async def membership(db,shop,user,owner=False):
    if user.role==UserRole.admin:return SimpleNamespace(id=None,role=ShopMemberRole.owner)
    if not approved(user.role,user.barber_verification_status):raise HTTPException(403,"เฉพาะบัญชีช่างที่แอดมินอนุมัติแล้วเท่านั้น")
    m=(await db.execute(select(ShopMember).where(ShopMember.shop_id==shop,ShopMember.user_id==user.id,ShopMember.is_active.is_(True)))).scalar_one_or_none()
    if not m or owner and m.role!=ShopMemberRole.owner:raise HTTPException(403,"คุณไม่มีสิทธิ์จัดการร้านนี้")
    return m
async def shop_or_404(db,id,active=False):
    q=select(Shop).where(Shop.id==id)
    if active:q=q.where(Shop.is_active.is_(True))
    s=(await db.execute(q)).scalar_one_or_none()
    if not s:raise HTTPException(404,"ไม่พบร้านตัดผม")
    return s
def staff_dict(m,u):return {"id":m.id,"user_id":m.user_id,"display_name":u.display_name,"role":m.role.value,"verification_status":u.barber_verification_status,"bio":m.bio,"specialties":m.specialties or [],"years_experience":m.years_experience,"is_active":m.is_active}
async def detail(db,s,user=None):
    services=list((await db.execute(select(ShopService).where(ShopService.shop_id==s.id).order_by(ShopService.is_active.desc(),ShopService.name))).scalars());rows=(await db.execute(select(ShopMember,User).join(User,User.id==ShopMember.user_id).where(ShopMember.shop_id==s.id))).all();hours=list((await db.execute(select(BusinessHour).where(BusinessHour.shop_id==s.id).order_by(BusinessHour.day_of_week))).scalars());m=None
    if user:m=(await db.execute(select(ShopMember).where(ShopMember.shop_id==s.id,ShopMember.user_id==user.id,ShopMember.is_active.is_(True)))).scalar_one_or_none()
    return {"id":s.id,"name":s.name,"description":s.description,"address":s.address,"latitude":s.latitude,"longitude":s.longitude,"phone":s.phone,"specialties":s.specialties or [],"timezone":s.timezone,"distance_km":None,"service_count":sum(x.is_active for x in services),"barber_count":sum(x.is_active for x,_ in rows),"services":services,"staff":[staff_dict(x,n) for x,n in rows],"business_hours":hours,"staff_role":"admin" if user and user.role==UserRole.admin else m.role.value if m else None}
@router.get("/shops",response_model=list[ShopItem])
async def list_shops(q:str|None=None,specialty:str|None=None,latitude:float|None=Query(None,ge=-90,le=90),longitude:float|None=Query(None,ge=-180,le=180),radius_km:float|None=Query(None,gt=0,le=200),db:AsyncSession=Depends(get_db)):
    if (latitude is None)!=(longitude is None):raise HTTPException(422,"ต้องส่งพิกัดให้ครบ")
    query=select(Shop).where(Shop.is_active.is_(True))
    if q:
        pattern=f"%{q.strip()}%";query=query.where(or_(Shop.name.ilike(pattern),Shop.address.ilike(pattern),Shop.description.ilike(pattern)))
    shops=list((await db.execute(query.limit(200))).scalars());needle=(specialty or "").strip().casefold()
    if needle:shops=[s for s in shops if any(needle in x.casefold() for x in (s.specialties or []))]
    out=[]
    for s in shops:
        d=haversine_km(latitude,longitude,s.latitude,s.longitude) if latitude is not None else None
        if radius_km and d is not None and d>radius_km:continue
        services=(await db.execute(select(func.count()).select_from(ShopService).where(ShopService.shop_id==s.id,ShopService.is_active.is_(True)))).scalar_one();barbers=(await db.execute(select(func.count()).select_from(ShopMember).join(User,User.id==ShopMember.user_id).where(ShopMember.shop_id==s.id,ShopMember.is_active.is_(True),User.role==UserRole.barber,User.barber_verification_status=="approved"))).scalar_one()
        out.append({"id":s.id,"name":s.name,"description":s.description,"address":s.address,"latitude":s.latitude,"longitude":s.longitude,"phone":s.phone,"specialties":s.specialties or [],"timezone":s.timezone,"distance_km":round(d,2) if d is not None else None,"service_count":services,"barber_count":barbers})
    return sorted(out,key=lambda x:(x["distance_km"] is None,x["distance_km"] or 0,x["name"]))[:100]
@router.post("/shops",response_model=ShopDetail,status_code=201)
async def create_shop(p:ShopCreate,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    if not can_create_shop(user.role,user.barber_verification_status):raise HTTPException(403,"ต้องเป็นบัญชีช่างที่แอดมินอนุมัติก่อนสร้างร้าน")
    if (await db.execute(select(ShopMember.id).where(ShopMember.user_id==user.id).limit(1))).scalar_one_or_none():raise HTTPException(409,"บัญชีช่างหนึ่งบัญชีสังกัดได้เพียงหนึ่งร้าน")
    try:ZoneInfo(p.timezone)
    except ZoneInfoNotFoundError:raise HTTPException(422,"เขตเวลาไม่ถูกต้อง")
    s=Shop(owner_user_id=user.id,**p.model_dump());db.add(s);await db.flush();db.add(ShopMember(shop_id=s.id,user_id=user.id,role=ShopMemberRole.owner,bio="เจ้าของร้าน",specialties=p.specialties,years_experience=user.barber_years_experience))
    for day in range(7):db.add(BusinessHour(shop_id=s.id,day_of_week=day,open_time=time(9) if day<6 else None,close_time=time(18) if day<6 else None,is_closed=day==6))
    try:await db.commit()
    except IntegrityError:await db.rollback();raise HTTPException(409,"บัญชีช่างหนึ่งบัญชีสร้างหรือสังกัดได้เพียงหนึ่งร้าน")
    await db.refresh(s);return await detail(db,s,user)
@router.get("/shops/mine",response_model=list[ShopItem])
async def my_shops(user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    if user.role==UserRole.admin:shops=list((await db.execute(select(Shop).order_by(Shop.name))).scalars())
    else:
        if not approved(user.role,user.barber_verification_status):raise HTTPException(403,"เฉพาะบัญชีช่างที่อนุมัติแล้วหรือแอดมินเท่านั้น")
        shops=list((await db.execute(select(Shop).join(ShopMember,ShopMember.shop_id==Shop.id).where(ShopMember.user_id==user.id,ShopMember.is_active.is_(True)))).scalars())
    return [{k:v for k,v in (await detail(db,s,user)).items() if k in ShopItem.model_fields} for s in shops]
@router.get("/shops/{shop_id}",response_model=ShopDetail)
async def get_shop(shop_id:uuid.UUID,user:User|None=Depends(get_current_user_optional),db:AsyncSession=Depends(get_db)):return await detail(db,await shop_or_404(db,shop_id,user is None),user)
@router.post("/shops/{shop_id}/staff",response_model=StaffOut,status_code=201)
async def add_staff(shop_id:uuid.UUID,p:StaffCreate,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    await membership(db,shop_id,user,True);u=(await db.execute(select(User).where(func.lower(User.email)==p.email.lower()))).scalar_one_or_none()
    if not u:raise HTTPException(404,"ไม่พบบัญชีช่างจากอีเมลนี้")
    if not approved(u.role,u.barber_verification_status):raise HTTPException(409,"บัญชีนี้ยังไม่ได้รับการอนุมัติเป็นช่าง")
    if (await db.execute(select(ShopMember.id).where(ShopMember.user_id==u.id).limit(1))).scalar_one_or_none():raise HTTPException(409,"บัญชีช่างนี้สร้างหรือสังกัดร้านอื่นอยู่แล้ว")
    m=ShopMember(shop_id=shop_id,user_id=u.id,role=ShopMemberRole.barber,bio=p.bio,specialties=p.specialties or u.barber_specialties,years_experience=p.years_experience or u.barber_years_experience);db.add(m)
    try:await db.commit()
    except IntegrityError:await db.rollback();raise HTTPException(409,"บัญชีช่างหนึ่งบัญชีสังกัดได้เพียงหนึ่งร้าน")
    await db.refresh(m);return staff_dict(m,u)
@router.post("/shops/{shop_id}/services",response_model=ServiceOut,status_code=201)
async def add_service(shop_id:uuid.UUID,p:ServiceCreate,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    await membership(db,shop_id,user,True);s=ShopService(shop_id=shop_id,**p.model_dump());db.add(s)
    try:await db.commit()
    except IntegrityError:await db.rollback();raise HTTPException(409,"มีบริการชื่อนี้แล้ว")
    await db.refresh(s);return s
@router.put("/shops/{shop_id}/business-hours",response_model=list[HourOut])
async def set_hours(shop_id:uuid.UUID,p:HoursSet,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    await membership(db,shop_id,user,True);rows={x.day_of_week:x for x in (await db.execute(select(BusinessHour).where(BusinessHour.shop_id==shop_id))).scalars()}
    for v in p.hours:
        row=rows.get(v.day_of_week)
        if not row:row=BusinessHour(shop_id=shop_id,day_of_week=v.day_of_week);db.add(row);rows[v.day_of_week]=row
        row.is_closed=v.is_closed;row.open_time=None if v.is_closed else v.open_time;row.close_time=None if v.is_closed else v.close_time
    await db.commit();return sorted(rows.values(),key=lambda x:x.day_of_week)
@router.get("/shops/{shop_id}/availability",response_model=list[SlotOut])
async def available(shop_id:uuid.UUID,target_date:date=Query(alias="date"),service_id:uuid.UUID=Query(),barber_member_id:uuid.UUID=Query(),db:AsyncSession=Depends(get_db)):
    s=await shop_or_404(db,shop_id,True);service=(await db.execute(select(ShopService).where(ShopService.id==service_id,ShopService.shop_id==shop_id,ShopService.is_active.is_(True)))).scalar_one_or_none();member=(await db.execute(select(ShopMember,User.display_name).join(User,User.id==ShopMember.user_id).where(ShopMember.id==barber_member_id,ShopMember.shop_id==shop_id,ShopMember.is_active.is_(True),User.role==UserRole.barber,User.barber_verification_status=="approved"))).one_or_none();hour=(await db.execute(select(BusinessHour).where(BusinessHour.shop_id==shop_id,BusinessHour.day_of_week==target_date.weekday()))).scalar_one_or_none()
    if not service or not member:raise HTTPException(404,"ไม่พบบริการหรือช่าง")
    if not hour or hour.is_closed:return []
    zone=ZoneInfo(s.timezone);start=datetime.combine(target_date,time.min,tzinfo=zone).astimezone(timezone.utc);end=start+timedelta(days=1);busy=[(b.starts_at,b.ends_at) for b in (await db.execute(select(Booking).where(Booking.barber_member_id==barber_member_id,Booking.status.in_(ACTIVE),Booking.starts_at<end,Booking.ends_at>start))).scalars()];m,name=member
    return [{"barber_member_id":m.id,"barber_name":name,"starts_at":a,"ends_at":b} for a,b in slots(target_date,hour.open_time,hour.close_time,s.timezone,service.duration_minutes,busy)][:100]
async def booking_row(db,id):
    c=aliased(User);bu=aliased(User);return (await db.execute(select(Booking,Shop.name,c.display_name,bu.display_name,ShopService.name,ShopService.price).join(Shop,Shop.id==Booking.shop_id).join(c,c.id==Booking.customer_id).join(ShopMember,ShopMember.id==Booking.barber_member_id).join(bu,bu.id==ShopMember.user_id).join(ShopService,ShopService.id==Booking.service_id).where(Booking.id==id))).one_or_none()
def booking_dict(row):
    b,sn,cn,bn,vn,price=row;return {"id":b.id,"shop_id":b.shop_id,"shop_name":sn,"customer_id":b.customer_id,"customer_name":cn,"barber_member_id":b.barber_member_id,"barber_name":bn,"service_id":b.service_id,"service_name":vn,"price":price,"starts_at":b.starts_at,"ends_at":b.ends_at,"status":b.status.value,"customer_note":b.customer_note,"cancel_reason":b.cancel_reason,"can_customer_cancel":b.status in ACTIVE and can_customer_cancel(b.starts_at),"created_at":b.created_at}
async def booking_out(db,id):
    row=await booking_row(db,id)
    if not row:raise HTTPException(404,"ไม่พบรายการจอง")
    return booking_dict(row)
async def booking_obj(db,id):
    b=(await db.execute(select(Booking).where(Booking.id==id))).scalar_one_or_none()
    if not b:raise HTTPException(404,"ไม่พบรายการจอง")
    return b
async def booking_staff(db,b,user):
    m=await membership(db,b.shop_id,user)
    if m.role==ShopMemberRole.barber and m.id!=b.barber_member_id:raise HTTPException(403,"ช่างจัดการได้เฉพาะคิวของตน")
@router.post("/bookings",response_model=BookingOut,status_code=201)
async def create_booking(p:BookingCreate,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    shop=await shop_or_404(db,p.shop_id,True);service=(await db.execute(select(ShopService).where(ShopService.id==p.service_id,ShopService.shop_id==shop.id,ShopService.is_active.is_(True)))).scalar_one_or_none();barber=(await db.execute(select(ShopMember).where(ShopMember.id==p.barber_member_id,ShopMember.shop_id==shop.id,ShopMember.is_active.is_(True)).with_for_update())).scalar_one_or_none()
    barber_user=(await db.execute(select(User).where(User.id==barber.user_id))).scalar_one_or_none() if barber else None
    if not service or not barber or not barber_user or not approved(barber_user.role,barber_user.barber_verification_status):raise HTTPException(404,"ไม่พบบริการหรือช่างที่ผ่านการอนุมัติ")
    start=p.starts_at.astimezone(timezone.utc);end=start+timedelta(minutes=service.duration_minutes)
    if start<=datetime.now(timezone.utc)+timedelta(minutes=5):raise HTTPException(422,"เวลานัดต้องอยู่ในอนาคต")
    local=start.astimezone(ZoneInfo(shop.timezone));local_end=end.astimezone(ZoneInfo(shop.timezone));hour=(await db.execute(select(BusinessHour).where(BusinessHour.shop_id==shop.id,BusinessHour.day_of_week==local.weekday()))).scalar_one_or_none()
    if local.date()!=local_end.date() or not hour or hour.is_closed or local.time().replace(tzinfo=None)<hour.open_time or local_end.time().replace(tzinfo=None)>hour.close_time:raise HTTPException(409,"เวลานี้อยู่นอกเวลาทำการ")
    taken=(await db.execute(select(Booking.id).where(Booking.barber_member_id==barber.id,Booking.status.in_(ACTIVE),Booking.starts_at<end,Booking.ends_at>start).limit(1))).scalar_one_or_none()
    if taken:raise HTTPException(409,"เวลานี้ถูกจองแล้ว")
    b=Booking(shop_id=shop.id,customer_id=user.id,barber_member_id=barber.id,service_id=service.id,starts_at=start,ends_at=end,status=BookingStatus.pending,customer_note=p.customer_note);db.add(b);await db.commit();await db.refresh(b);return await booking_out(db,b.id)
@router.get("/bookings/mine",response_model=list[BookingOut])
async def mine(user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    ids=list((await db.execute(select(Booking.id).where(Booking.customer_id==user.id).order_by(Booking.starts_at.desc()).limit(100))).scalars());return [await booking_out(db,x) for x in ids]
@router.get("/bookings/shop/{shop_id}",response_model=list[BookingOut])
async def shop_bookings(shop_id:uuid.UUID,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    m=await membership(db,shop_id,user);q=select(Booking.id).where(Booking.shop_id==shop_id)
    if m.role==ShopMemberRole.barber:q=q.where(Booking.barber_member_id==m.id)
    ids=list((await db.execute(q.order_by(Booking.starts_at).limit(200))).scalars());return [await booking_out(db,x) for x in ids]
async def change(id,action,reason,user,db):
    b=await booking_obj(db,id)
    if action=="cancel" and b.customer_id==user.id:
        if b.status not in ACTIVE or not can_customer_cancel(b.starts_at):raise HTTPException(409,"ลูกค้าต้องยกเลิกล่วงหน้าอย่างน้อย 6 ชั่วโมง")
    else:await booking_staff(db,b,user)
    if action=="confirm":
        if b.status!=BookingStatus.pending:raise HTTPException(409,"สถานะไม่ถูกต้อง")
        b.status=BookingStatus.confirmed
    elif action=="reject":
        if b.status!=BookingStatus.pending:raise HTTPException(409,"สถานะไม่ถูกต้อง")
        b.status=BookingStatus.rejected
    elif action=="cancel":
        if b.status not in ACTIVE:raise HTTPException(409,"ยกเลิกไม่ได้")
        b.status=BookingStatus.cancelled
    elif action=="complete":
        if b.status!=BookingStatus.confirmed:raise HTTPException(409,"สถานะไม่ถูกต้อง")
        b.status=BookingStatus.completed
    b.cancel_reason=reason if action in ("reject","cancel") else b.cancel_reason;b.cancelled_by_id=user.id if action in ("reject","cancel") else b.cancelled_by_id;await db.commit();return await booking_out(db,b.id)
@router.post("/bookings/{id}/confirm",response_model=BookingOut)
async def confirm(id:uuid.UUID,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):return await change(id,"confirm",None,user,db)
@router.post("/bookings/{id}/reject",response_model=BookingOut)
async def reject(id:uuid.UUID,p:Action,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):return await change(id,"reject",p.reason or "ร้านไม่สามารถรับคิวนี้ได้",user,db)
@router.post("/bookings/{id}/cancel",response_model=BookingOut)
async def cancel(id:uuid.UUID,p:Action,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):return await change(id,"cancel",p.reason or "ยกเลิกการจอง",user,db)
@router.post("/bookings/{id}/complete",response_model=BookingOut)
async def complete(id:uuid.UUID,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):return await change(id,"complete",None,user,db)
class Manager:
    def __init__(self):self.rooms=defaultdict(set);self.lock=asyncio.Lock()
    async def connect(self,room,ws):await ws.accept();self.rooms[room].add(ws)
    async def disconnect(self,room,ws):
        self.rooms[room].discard(ws)
        if not self.rooms[room]:self.rooms.pop(room,None)
    async def send(self,room,data):
        for ws in list(self.rooms.get(room,set())):
            try:await ws.send_json(jsonable_encoder(data))
            except:await self.disconnect(room,ws)
manager=Manager()
async def chat_access(db,shop,customer,user):
    if user.id==customer:await shop_or_404(db,shop,True);return None
    m=await membership(db,shop,user)
    if m.role==ShopMemberRole.barber and not (await db.execute(select(Booking.id).where(Booking.shop_id==shop,Booking.customer_id==customer,Booking.barber_member_id==m.id).limit(1))).scalar_one_or_none():raise HTTPException(403,"ช่างเปิดได้เฉพาะแชตลูกค้าที่จองกับตน")
    return m
async def messages(db,shop,customer):
    rows=list((await db.execute(select(ShopMessage,User.display_name).join(User,User.id==ShopMessage.sender_id).where(ShopMessage.shop_id==shop,ShopMessage.customer_id==customer).order_by(ShopMessage.created_at.desc()).limit(200))).all());staff=set((await db.execute(select(ShopMember.user_id).where(ShopMember.shop_id==shop,ShopMember.is_active.is_(True)))).scalars());return [{"id":m.id,"shop_id":m.shop_id,"customer_id":m.customer_id,"sender_id":m.sender_id,"sender_name":n,"sender_is_staff":m.sender_id in staff,"body":m.body,"created_at":m.created_at} for m,n in reversed(rows)]
@router.get("/shop-chat/shops/{shop}/conversations",response_model=list[ConversationOut])
async def convos(shop:uuid.UUID,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    m=await membership(db,shop,user);c=aliased(User);q=select(ShopMessage,c.display_name).join(c,c.id==ShopMessage.customer_id).where(ShopMessage.shop_id==shop)
    if user.role!=UserRole.admin and m.role==ShopMemberRole.barber:q=q.where(ShopMessage.customer_id.in_(select(Booking.customer_id).where(Booking.barber_member_id==m.id)))
    rows=(await db.execute(q.order_by(ShopMessage.created_at.desc()).limit(1000))).all();seen=set();out=[]
    for msg,n in rows:
        if msg.customer_id in seen:continue
        seen.add(msg.customer_id);out.append({"customer_id":msg.customer_id,"customer_name":n,"last_message":msg.body,"last_message_at":msg.created_at})
    return out[:100]
@router.get("/shop-chat/shops/{shop}/{customer}",response_model=list[MessageOut])
async def get_messages(shop:uuid.UUID,customer:uuid.UUID,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):await chat_access(db,shop,customer,user);return await messages(db,shop,customer)
@router.post("/shop-chat/shops/{shop}/{customer}",response_model=MessageOut)
async def post_message(shop:uuid.UUID,customer:uuid.UUID,p:MessageCreate,user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    await chat_access(db,shop,customer,user);m=ShopMessage(shop_id=shop,customer_id=customer,sender_id=user.id,body=p.body);db.add(m);await db.commit();result=(await messages(db,shop,customer))[-1];await manager.send((shop,customer),{"type":"message","message":result});return result
@router.websocket("/shop-chat/ws/{shop}/{customer}")
async def socket(ws:WebSocket,shop:uuid.UUID,customer:uuid.UUID):
    claims=decode_access_token(ws.query_params.get("token")) if ws.query_params.get("token") else None
    try:uid=uuid.UUID(claims["sub"]) if claims else None
    except:uid=None
    if not uid:await ws.close(code=4401);return
    room=(shop,customer)
    async with AsyncSessionLocal() as db:
        user=(await db.execute(select(User).where(User.id==uid))).scalar_one_or_none()
        if not user:await ws.close(code=4401);return
        try:await chat_access(db,shop,customer,user)
        except HTTPException:await ws.close(code=4403);return
        await manager.connect(room,ws);await ws.send_json(jsonable_encoder({"type":"history","messages":await messages(db,shop,customer)}))
        try:
            while True:
                raw=await ws.receive_json()
                try:body=MessageCreate(body=raw.get("body","")).body
                except ValidationError:await ws.send_json({"type":"error","message":"ข้อความไม่ถูกต้อง"});continue
                m=ShopMessage(shop_id=shop,customer_id=customer,sender_id=user.id,body=body);db.add(m);await db.commit();result=(await messages(db,shop,customer))[-1];await manager.send(room,{"type":"message","message":result})
        except WebSocketDisconnect:pass
        finally:await manager.disconnect(room,ws)
