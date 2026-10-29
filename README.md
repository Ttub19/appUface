# Face Shape AI — Full Stack (Frontend + Backend + PostgreSQL)

โปรเจกต์เดียวที่รวม **frontend** (HTML/Tailwind/Vanilla JS + MediaPipe Face Mesh), **backend** (FastAPI, async), และ **database** (PostgreSQL) เข้าด้วยกัน — frontend เรียก API จริงแล้ว ไม่ใช่ mock อีกต่อไป

## ภาพรวมสถาปัตยกรรม

```
Browser ──▶ nginx (web:80, expose 5173) ──▶ /api/* proxy ──▶ FastAPI (api:8000) ──▶ PostgreSQL (db:5432)
              │
              └──▶ ไฟล์ static (index.html, css, js) เสิร์ฟตรงจาก nginx
```

- Browser เรียก `fetch('/api/...')` ไปที่ nginx (origin เดียวกับหน้าเว็บ) → nginx proxy ต่อไปยัง FastAPI container → ไม่มีปัญหา CORS เลยในโหมด docker-compose
- MediaPipe Face Mesh (ติดตามใบหน้า/คำนวณ Golden Ratio/AR แว่น-ทรงผม) ยังทำงาน **ฝั่ง client ทั้งหมด** เหมือนเดิม ไม่ได้ผ่าน backend — backend มีหน้าที่ auth, เก็บผลลัพธ์ (history/favorites/golden-ratio snapshot), และรันโมเดลจริง (EfficientNetB1, `backend/models/`) สำหรับ face-shape classification

## โครงสร้างโปรเจกต์

```
face-shape-app/
├── docker-compose.yml     # รวม postgres + backend + frontend เป็นคำสั่งเดียว
├── .env.example            # ค่า config ของทั้ง 3 services (copy เป็น .env)
├── backend/                 # FastAPI + SQLAlchemy (async) — ดู backend/README.md
└── frontend/                 # HTML/Tailwind/Vanilla JS ES Modules — ดู frontend/README.md
```

## วิธีรัน (แนะนำ: Docker Compose)

```bash
cp .env.example .env
# แก้ JWT_SECRET_KEY ใน .env ให้เป็นค่าสุ่มจริง:
python3 -c "import secrets; print(secrets.token_urlsafe(48))"

docker compose up --build
```

จากนั้นเปิด:
- **หน้าเว็บ**: http://localhost:5173
- **API docs (Swagger)**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432 (user/password/db ตามที่ตั้งใน `.env`)

ตารางในฐานข้อมูลถูกสร้างอัตโนมัติตอน backend เริ่มทำงาน (ดู `backend/sql/schema.sql` สำหรับ DDL อ้างอิง)

## ใช้งานจริง

1. เปิด http://localhost:5173 → หน้า login
2. กด **"สมัครสมาชิก"** เพื่อสร้างบัญชีใหม่ (หรือกด **"เข้าใช้งานแบบผู้เยี่ยมชม"** เพื่อดูผลวิเคราะห์ได้โดยไม่ต้องมีบัญชี แต่บันทึกอะไรไม่ได้)
3. หลัง login แล้วใช้งานได้ทุกฟีเจอร์: สแกน/วิเคราะห์หน้า (บันทึกจริงลง PostgreSQL), Golden Ratio Analysis, ลองแว่น/ทรงผมเสมือนจริงผ่านกล้อง, รายการโปรด, ประวัติ (ลบ/ค้นหา/export CSV จริง), โปรไฟล์ + ธีมสี (บันทึกจริง)
4. ถ้าต้องการดู Admin dashboard: ต้องตั้ง role ผู้ใช้เป็น `admin` ในฐานข้อมูลเอง (ไม่มี UI ให้เปลี่ยน role เพื่อความปลอดภัย) — ดูวิธีด้านล่าง

### ตั้งให้บัญชีหนึ่งเป็น admin

```bash
docker compose exec db psql -U faceshape -d faceshape \
  -c "UPDATE users SET role = 'admin' WHERE email = 'your@email.com';"
```

หรือบน Render: ใส่ Environment `ADMIN_EMAILS=your@email.com` แล้ว deploy (ไม่ต้องต่อฐานข้อมูลเอง) / หรือรัน `python backend/scripts/set_admin.py` ด้วย External Database URL

จากนั้น logout แล้ว login ใหม่อีกครั้ง จะเห็นเมนู "Dashboard" (Admin Insights)

## รันแบบไม่ใช้ Docker (แยกรันเอง)

ดูรายละเอียดใน `backend/README.md` (รัน FastAPI + PostgreSQL เอง) และ `frontend/README.md` (รัน static server เอง) — กรณีนี้ frontend เรียก backend คนละ origin จึงต้องตั้ง `CORS_ORIGINS` ในฝั่ง backend ให้ตรงกับ origin ของ frontend และตั้ง `window.__API_BASE_URL__` ในฝั่ง frontend ให้ชี้ไปที่ backend ตรง ๆ (ดูคอมเมนต์ใน `frontend/js/apiClient.js`)

## หมายเหตุด้านความปลอดภัย
- รหัสผ่าน hash ด้วย bcrypt, ไม่มี secret hardcode ในโค้ด (อ่านจาก `.env` ทั้งหมด)
- JWT เก็บใน `localStorage` ฝั่ง browser (มาตรฐานสำหรับเว็บแอปทั่วไป — ถ้าต้องการความปลอดภัยสูงขึ้นสำหรับงานจริง ควรพิจารณาใช้ httpOnly cookie แทน)
- รูปภาพที่อัปโหลด/ถ่ายจากกล้องเก็บเป็น base64 ตรงในฐานข้อมูล — เหมาะสำหรับ demo เท่านั้น งานจริงควรอัปโหลดขึ้น object storage (S3/GCS) แล้วเก็บแค่ URL
- ไม่มี endpoint ให้ตั้ง role เป็น admin เองผ่าน API (ต้องแก้ตรงในฐานข้อมูล) เพื่อป้องกันการยกระดับสิทธิ์


## Barber shop booking
See `BOOKING_SYSTEM.md`. The original member/admin roles are preserved; shop ownership and staff are separate memberships.
