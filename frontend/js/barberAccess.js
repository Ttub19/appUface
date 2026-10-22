export const approvedBarber=u=>u?.role==="barber"&&u?.barber_verification_status==="approved";
export const canManageShops=u=>u?.role==="admin"||approvedBarber(u);
export const canCreateShop=u=>approvedBarber(u);
export const barberStatusMeta=s=>({pending:{label:"รอแอดมินตรวจสอบ",tone:"warning"},approved:{label:"อนุมัติเป็นช่างแล้ว",tone:"success"},rejected:{label:"ไม่ผ่านการอนุมัติ",tone:"danger"},none:{label:"บัญชีสมาชิกทั่วไป",tone:"neutral"}}[s]||{label:"ไม่ทราบสถานะ",tone:"neutral"});
