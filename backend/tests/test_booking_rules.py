import unittest
from datetime import date,datetime,time,timedelta,timezone
from app.services.booking_rules import can_customer_cancel,haversine_km,overlap,slots
class Tests(unittest.TestCase):
 def test_distance(self):self.assertTrue(.5<haversine_km(13.7563,100.5018,13.7466,100.5347)<5)
 def test_cancel(self):
  n=datetime(2026,9,17,10,tzinfo=timezone.utc);self.assertTrue(can_customer_cancel(n+timedelta(hours=6),n));self.assertFalse(can_customer_cancel(n+timedelta(hours=5),n))
 def test_overlap(self):
  n=datetime(2026,9,18,2,tzinfo=timezone.utc);self.assertFalse(overlap(n,n+timedelta(hours=1),n+timedelta(hours=1),n+timedelta(hours=2)))
 def test_slots(self):
  n=datetime(2026,9,17,1,tzinfo=timezone.utc);busy=[(datetime(2026,9,18,3,tzinfo=timezone.utc),datetime(2026,9,18,4,tzinfo=timezone.utc))];self.assertEqual([x[0].strftime("%H:%M") for x in slots(date(2026,9,18),time(9),time(12),"Asia/Bangkok",60,busy,now=n)],["09:00","11:00"])
if __name__=="__main__":unittest.main()
