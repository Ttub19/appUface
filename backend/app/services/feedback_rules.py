"""Pure rules for the feedback / contact-admin form (no DB, unit-tested)."""
from datetime import timedelta

RATE_LIMIT_COUNT = 5                      # messages ...
RATE_LIMIT_WINDOW = timedelta(minutes=10)  # ... per sender per 10 minutes


def resolve_contact(user, contact_name, contact_email):
    """
    Who the admin should reply to. Signed-in users: their account name/email
    (a name typed in the form wins, the email always comes from the account).
    Guests must give both. Returns (name, email); raises ValueError otherwise.
    """
    name = (contact_name or "").strip()
    if user is not None:
        return (name or user.display_name)[:120], user.email
    email = (str(contact_email) if contact_email else "").strip().lower()
    if not name or not email:
        raise ValueError("ผู้ใช้ทั่วไปต้องกรอกชื่อและอีเมลเพื่อให้แอดมินติดต่อกลับได้")
    return name[:120], email


def handled_fields(status, admin_id, now):
    """handled_by/at follow the status: set when an admin reads or resolves, cleared on 'new'."""
    if status == "new":
        return None, None
    return admin_id, now
