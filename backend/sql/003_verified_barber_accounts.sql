-- Run after 002_booking_system.sql on an existing PostgreSQL database.
ALTER TYPE user_role ADD VALUE IF NOT EXISTS 'barber';
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_verification_status VARCHAR(16) NOT NULL DEFAULT 'none';
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_legal_name VARCHAR(160);
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_phone VARCHAR(40);
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_years_experience INTEGER NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_specialties JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_application_note TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_review_note TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_reviewed_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS barber_reviewed_by_id UUID;
UPDATE users u SET role='barber'::user_role,barber_verification_status='approved',barber_legal_name=COALESCE(u.barber_legal_name,u.display_name),barber_review_note=COALESCE(u.barber_review_note,'แปลงสิทธิ์จากเจ้าของร้านเดิมอัตโนมัติ'),barber_reviewed_at=COALESCE(u.barber_reviewed_at,now()) WHERE EXISTS(SELECT 1 FROM barber_shops s WHERE s.owner_user_id=u.id);
UPDATE users u SET barber_verification_status='pending',barber_legal_name=COALESCE(u.barber_legal_name,u.display_name),barber_application_note=COALESCE(u.barber_application_note,'บัญชีช่างเดิม กรุณาตรวจสอบก่อนเปิดสิทธิ์') WHERE u.role='member'::user_role AND u.barber_verification_status='none' AND EXISTS(SELECT 1 FROM shop_members sm WHERE sm.user_id=u.id AND sm.role='barber'::shop_member_role);
DO $$ BEGIN IF NOT EXISTS(SELECT 1 FROM shop_members GROUP BY user_id HAVING count(*)>1) THEN CREATE UNIQUE INDEX IF NOT EXISTS uq_shop_member_one_shop_idx ON shop_members(user_id); END IF; END $$;
