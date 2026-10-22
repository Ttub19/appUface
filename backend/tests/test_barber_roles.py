import unittest
from app.services.barber_roles import approved,can_administer,can_create_shop,can_manage_booking
class Tests(unittest.TestCase):
 def test_pending(self):self.assertFalse(approved("member","pending"));self.assertFalse(can_create_shop("member","pending"))
 def test_approved(self):self.assertTrue(can_create_shop("barber","approved"));self.assertFalse(can_create_shop("admin","none"))
 def test_owner_admin(self):self.assertTrue(can_administer("barber","approved","owner"));self.assertTrue(can_administer("admin","none",None));self.assertFalse(can_administer("barber","approved","barber"))
 def test_own_queue(self):self.assertTrue(can_manage_booking("barber","approved","barber","1","1"));self.assertFalse(can_manage_booking("barber","approved","barber","1","2"));self.assertTrue(can_manage_booking("barber","approved","owner","1","2"))
if __name__=="__main__":unittest.main()
