# Face Shape AI — Backend (FastAPI + PostgreSQL)

Backend API สำหรับโปรเจกต์ Face Shape AI มาแทนที่ mock functions ฝั่ง client (`FaceShapeApiClient`, `localStorage`-style state ฯลฯ) ด้วย REST API จริงที่ต่อกับ PostgreSQL

## เทคโนโลยีที่ใช้
- **FastAPI** (async) + **Uvicorn**
- **SQLAlchemy 2.0** (async engine ผ่าน `asyncpg`) — ORM models ใน `app/models.py`
- **PostgreSQL**
- **JWT** (python-jose) สำหรับ auth + **passlib/bcrypt** สำหรับ hash รหัสผ่าน
- Config ทั้งหมดอ่านจาก environment variables (`pydantic-settings`) — ไม่มี secret ฝังในโค้ด

## โครงสร้างโปรเจกต์

```
face-shape-backend/
├── app/
│   ├── main.py                        # สร้าง FastAPI app, CORS, รวม router, สร้างตารางตอน startup
│   ├── config.py                       # อ่านค่า config จาก .env (DB URL, JWT secret, CORS origins)
│   ├── database.py                     # SQLAlchemy async engine/session + Base
│   ├── models.py                       # ORM models: User, ScanHistory, Favorite, GoldenRatioResult
│   ├── schemas.py                       # Pydantic request/response schemas
│   ├── security.py                      # hash รหัสผ่าน + สร้าง/ตรวจสอบ JWT
│   ├── deps.py                           # dependency: get_current_user, require_role (RBAC)
│   ├── routers/
│   │   ├── auth.py                       # POST /auth/register, /auth/login, GET /auth/me
│   │   ├── scans.py                       # POST /scans/predict, POST/GET /scans, DELETE /scans/{id}, GET /scans/export.csv
│   │   ├── favorites.py                    # POST/GET /favorites, DELETE /favorites/{id}
│   │   ├── golden_ratio.py                  # POST/GET /golden-ratio
│   │   ├── profile.py                        # GET/PUT /profile
│   │   └── admin.py                           # GET /admin/stats (admin เท่านั้น)
│   └── services/
│       ├── face_shape_predictor.py            # โมเดลจริง EfficientNetB1 (test 80.4%) + mock สำหรับ dev (FACE_SHAPE_PREDICTOR=mock)
│       └── face_crop.py                       # ครอปหน้าด้วย MediaPipe แบบเดียวกับตอนเทรน
├── sql/schema.sql                       # DDL อ้างอิง (แอปสร้างตารางเองตอน startup ผ่าน SQLAlchemy)
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

> หมายเหตุ: `docker-compose.yml` ที่รวม PostgreSQL + backend + frontend อยู่ที่ **root ของโปรเจกต์** (`../docker-compose.yml`) ไม่ใช่ในโฟลเดอร์นี้ — ดู `../README.md` สำหรับวิธีรันทั้งระบบ

## วิธีรัน

### แบบเร็วที่สุด: รันทั้งระบบจาก root (แนะนำ)

```bash
cd ..                      # กลับไปที่ root ของโปรเจกต์ (face-shape-app/)
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # ใส่ผลลัพธ์ลงใน JWT_SECRET_KEY ของ .env
docker compose up --build
```

จะได้ PostgreSQL รันที่ `localhost:5432`, API รันที่ `http://localhost:8000` (Swagger UI ที่ `http://localhost:8000/docs`), และหน้าเว็บที่ `http://localhost:5173`

### แบบรันเองโดยไม่ใช้ Docker

```bash
# 1) ติดตั้ง PostgreSQL แล้วสร้าง database (ปรับชื่อ/รหัสผ่านตามต้องการ)
createdb faceshape

# 2) ติดตั้ง dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3) ตั้งค่า .env
cp .env.example .env
# แก้ DATABASE_URL ให้ตรงกับ PostgreSQL ของคุณ และตั้ง JWT_SECRET_KEY เป็นค่าสุ่มจริง

# 4) รัน
uvicorn app.main:app --reload --port 8000
```

### รันเองบน Windows (Command Prompt)

```bat
py -m venv venv
venv\Scripts\activate
python -m uvicorn app.main:app --reload --port 8000
```

แก้ค่า `DATABASE_URL` ในไฟล์ `.env` ให้ตรงกับ PostgreSQL ก่อนรัน หากใช้ PowerShell ให้เปิดใช้งานด้วย `venv\Scripts\Activate.ps1` แทน `venv\Scripts\activate`

ตารางทั้งหมดจะถูกสร้างอัตโนมัติตอน startup (ดู `sql/schema.sql` สำหรับ DDL อ้างอิงถ้าต้องการสร้างมือหรือย้ายไปใช้ Alembic migration ในอนาคต)

## API overview

| Method | Path | ต้อง login? | คำอธิบาย |
|---|---|---|---|
| POST | `/auth/register` | ไม่ | สร้างบัญชีสมาชิก |
| POST | `/auth/login` | ไม่ | login (OAuth2 password flow) → คืน JWT |
| GET | `/auth/me` | ต้อง | ข้อมูลผู้ใช้ปัจจุบัน |
| POST | `/scans/predict` | ไม่ (guest ก็เรียกได้) | วิเคราะห์รูปทรงหน้าจากรูปด้วยโมเดลจริง (400 = ไม่พบใบหน้า, 503 = โมเดลยังไม่พร้อม) |
| POST | `/scans` | ต้อง | บันทึกผลสแกนลงประวัติ |
| GET | `/scans` | ต้อง | ดึงประวัติ (query: `q`, `shape`) |
| DELETE | `/scans/{id}` | ต้อง | ลบประวัติ 1 รายการ |
| GET | `/scans/export.csv` | ต้อง | ดาวน์โหลดประวัติเป็น CSV |
| POST | `/favorites` | ต้อง | บันทึกรายการโปรด |
| GET | `/favorites` | ต้อง | ดึงรายการโปรด |
| DELETE | `/favorites/{id}` | ต้อง | ลบรายการโปรด |
| POST | `/golden-ratio` | ต้อง | บันทึกผลวิเคราะห์ Golden Ratio |
| GET | `/golden-ratio` | ต้อง | ดึงประวัติผลวิเคราะห์ Golden Ratio |
| GET/PUT | `/profile` | ต้อง | ดู/แก้ไขโปรไฟล์และธีมสี |
| GET | `/admin/stats` | ต้อง (admin) | สถิติสำหรับ Admin dashboard |

Auth ใช้ `Authorization: Bearer <token>` header หลัง login (มาตรฐาน OAuth2 password flow — ลอง `/docs` แล้วกด "Authorize" ได้เลย)

## บทบาทผู้ใช้ (roles)

- **guest** (ฝั่งหน้าเว็บ) = ไม่ได้ส่ง token มาเลย → เรียกได้แค่ `/scans/predict` (ดูผลได้ แต่บันทึกไม่ได้ ตรงกับ mockup เดิม)
- **member** = สมัคร/login แล้ว → ใช้ทุกฟีเจอร์ยกเว้น admin stats
- **admin** = ตั้งค่า role ในฐานข้อมูลเป็น `admin` (ตอนนี้ยังไม่มี endpoint เปลี่ยน role เอง เพื่อความปลอดภัย — ปรับตรงในฐานข้อมูลหรือเพิ่ม endpoint ที่มีการป้องกันแน่นหนากว่านี้ตามต้องการ)

## หมายเหตุด้านความปลอดภัย
- รหัสผ่าน hash ด้วย bcrypt ก่อนเก็บ ไม่มีการเก็บ plaintext
- JWT secret / DB credentials ต้องมาจาก `.env` เท่านั้น (ไม่มี hardcode ในโค้ด, `.env` ไม่ควร commit เข้า git)
- CORS จำกัดเฉพาะ origin ที่อยู่ใน `CORS_ORIGINS`
- การเก็บรูปภาพเป็น base64 ตรงในฐานข้อมูล (คอลัมน์ `image_data_url`) เหมาะสำหรับ demo เท่านั้น — งานจริงควรอัปโหลดขึ้น object storage (S3/GCS) แล้วเก็บแค่ URL

## เชื่อมกับ Front-end

โปรเจกต์นี้ถูกรวมกับ front-end แล้ว — ดู `../README.md` (root) สำหรับวิธีรันทั้งระบบด้วย `docker compose up` คำสั่งเดียว (PostgreSQL + backend นี้ + frontend ผ่าน nginx reverse proxy) Front-end เรียก endpoint เหล่านี้ทั้งหมดจริงแล้ว ไม่ใช่ mock อีกต่อไป

## โมเดลวิเคราะห์รูปหน้า (`models/`)

- `face_shape_effnetb1.keras.part00..02` + `.sha256` — โมเดล EfficientNetB1 (Keras 3.13) แบ่งเป็นไฟล์ย่อยไม่เกิน 25 MB
  ตอนรันครั้งแรก backend จะรวมเป็น `face_shape_effnetb1.keras` ให้เองและเช็ก SHA-256 (ไฟล์ที่รวมแล้วอยู่ใน `.gitignore`)
- `class_names.json` — ลำดับคลาสและค่าการครอปหน้าที่ใช้ตอนเทรน (ต้องตรงกับ `app/services/face_crop.py`)
- `face_landmarker.task` — โมเดลหาใบหน้าของ MediaPipe (~3.6 MB) ดาวน์โหลดอัตโนมัติครั้งแรก ต้องต่ออินเทอร์เน็ต
- โหลดโมเดลเบื้องหลังตอนเปิดเซิร์ฟเวอร์ (ใช้เวลาหลายวินาที) — log `Face-shape model ready` = พร้อมใช้
- ไม่มี TensorFlow ในเครื่อง dev? ตั้ง `FACE_SHAPE_PREDICTOR=mock` เพื่อใช้ผลสุ่มแบบเดิมชั่วคราว
- เปลี่ยนโมเดลใหม่: วางไฟล์ `.keras` ทับ `face_shape_effnetb1.keras` แล้วลบ `.sha256` (หรือแบ่ง part ใหม่พร้อม checksum) และอัปเดต `class_names.json`
