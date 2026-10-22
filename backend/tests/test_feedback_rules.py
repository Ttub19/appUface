import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from app.services.feedback_rules import handled_fields, resolve_contact


class ResolveContact(unittest.TestCase):
    user = SimpleNamespace(display_name="สมชาย", email="somchai@example.com")

    def test_member_uses_account(self):
        self.assertEqual(resolve_contact(self.user, None, None), ("สมชาย", "somchai@example.com"))

    def test_member_cannot_spoof_email_but_can_set_name(self):
        self.assertEqual(resolve_contact(self.user, " ชาย ", "other@example.com"), ("ชาย", "somchai@example.com"))

    def test_guest_needs_name_and_email(self):
        with self.assertRaises(ValueError):
            resolve_contact(None, "", "a@b.co")
        with self.assertRaises(ValueError):
            resolve_contact(None, "เอ", None)

    def test_guest_email_normalised(self):
        self.assertEqual(resolve_contact(None, "เอ", " A@B.CO "), ("เอ", "a@b.co"))


class HandledFields(unittest.TestCase):
    def test_new_clears_and_others_set(self):
        now = datetime.now(timezone.utc)
        self.assertEqual(handled_fields("new", "x", now), (None, None))
        self.assertEqual(handled_fields("resolved", "x", now), ("x", now))


if __name__ == "__main__":
    unittest.main()
