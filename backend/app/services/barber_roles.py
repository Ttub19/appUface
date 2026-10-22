"""Pure role rules shared by booking authorization and unit tests."""
def value(v): return str(getattr(v,"value",v))
def approved(role,status): return value(role)=="barber" and value(status)=="approved"
def can_create_shop(role,status): return approved(role,status)
def can_administer(role,status,shop_role): return value(role)=="admin" or (approved(role,status) and value(shop_role)=="owner")
def can_manage_booking(role,status,shop_role,member_id,assigned_id):
    if value(role)=="admin": return True
    if not approved(role,status): return False
    return value(shop_role)=="owner" or (value(shop_role)=="barber" and str(member_id)==str(assigned_id))
