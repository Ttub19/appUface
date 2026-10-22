from datetime import date,datetime,time,timedelta,timezone
from math import asin,cos,radians,sin,sqrt
from zoneinfo import ZoneInfo
def haversine_km(a,b,c,d):
    a,b,c,d=map(radians,(a,b,c,d));v=sin((c-a)/2)**2+cos(a)*cos(c)*sin((d-b)/2)**2;return 2*6371.0088*asin(sqrt(v))
def utc(v):
    if v.tzinfo is None:raise ValueError("timezone required")
    return v.astimezone(timezone.utc)
def can_customer_cancel(starts,now=None,hours=6):return utc(starts)-utc(now or datetime.now(timezone.utc))>=timedelta(hours=hours)
def overlap(a,b,c,d):return utc(a)<utc(d) and utc(b)>utc(c)
def slots(day:date,opening:time,closing:time,tzname:str,duration:int,busy,now=None,step=30):
    zone=ZoneInfo(tzname);cur=datetime.combine(day,opening,tzinfo=zone);end=datetime.combine(day,closing,tzinfo=zone);length=timedelta(minutes=duration);inc=timedelta(minutes=step);ref=utc(now or datetime.now(timezone.utc));out=[]
    while cur+length<=end:
        finish=cur+length
        if utc(cur)>ref and not any(overlap(cur,finish,x,y) for x,y in busy):out.append((cur,finish))
        cur+=inc
    return out
