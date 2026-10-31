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
│       ├── face_shape_predictor.py            # ResNet50 แยกเพศ ensemble 3 fold + TTA บน LiteRT + mock สำหรับ dev
│       ├── image_ops.py                       # resize/preprocess แบบ NumPy (ตรงกับ TensorFlow ตอนเทรน)
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
- **admin** = ใส่อีเมลใน env `ADMIN_EMAILS` (คั่นด้วย `,`) แล้ว deploy/รีสตาร์ต — บัญชีที่มีอยู่จะถูกตั้งเป็น admin ตอนเซิร์ฟเวอร์เริ่ม และตอน login (ใส่เฉพาะอีเมลที่สมัครไว้แล้ว เพราะไม่มีการยืนยันอีเมล) หรือแก้ตรงในฐานข้อมูลด้วย `python backend/scripts/set_admin.py`

## หมายเหตุด้านความปลอดภัย
- รหัสผ่าน hash ด้วย bcrypt ก่อนเก็บ ไม่มีการเก็บ plaintext
- JWT secret / DB credentials ต้องมาจาก `.env` เท่านั้น (ไม่มี hardcode ในโค้ด, `.env` ไม่ควร commit เข้า git)
- CORS จำกัดเฉพาะ origin ที่อยู่ใน `CORS_ORIGINS`
- การเก็บรูปภาพเป็น base64 ตรงในฐานข้อมูล (คอลัมน์ `image_data_url`) เหมาะสำหรับ demo เท่านั้น — งานจริงควรอัปโหลดขึ้น object storage (S3/GCS) แล้วเก็บแค่ URL

## เชื่อมกับ Front-end

โปรเจกต์นี้ถูกรวมกับ front-end แล้ว — ดู `../README.md` (root) สำหรับวิธีรันทั้งระบบด้วย `docker compose up` คำสั่งเดียว (PostgreSQL + backend นี้ + frontend ผ่าน nginx reverse proxy) Front-end เรียก endpoint เหล่านี้ทั้งหมดจริงแล้ว ไม่ใช่ mock อีกต่อไป

## โมเดลวิเคราะห์รูปหน้า (`models/`)

- `face_shape_<female|male>_fold1..3.tflite.part*` + `.sha256` — ResNet50 แยกเพศ ครบ 3 fold (training v3) แปลงเป็น TFLite
  (น้ำหนัก int8) ด้วย `training/convert_tflite.py` ตอนรันครั้งแรก backend จะรวม part เป็นไฟล์และเช็ก SHA-256 ให้เอง
- ทายแบบ **ensemble 3 fold + TTA (กลับซ้าย-ขวา)**: แต่ละรูปผ่านโมเดล 3 ตัว x 2 ภาพ แล้วเฉลี่ยความน่าจะเป็น
  (ชุดทดสอบเดิม: หญิง top-1 80.8% / top-2 91.2%, ชาย top-1 85.9% / top-2 97.0%) — API ส่ง `ranking` ทุกทรงเรียงจากมากไปน้อย
  หน้าเว็บแสดงทรงที่ใกล้เคียงรองลงมาด้วย
- รันบน **LiteRT** (`ai-edge-litert`) ไม่ต้องติดตั้ง TensorFlow — RAM น้อยพอสำหรับ Render ฟรี (512 MB)
- `class_names.json` — ลำดับคลาส ไฟล์โมเดลของแต่ละเพศ (`models`), `"tta": "flip"` และค่าการครอปหน้าที่ใช้ตอนเทรน (ต้องตรงกับ `app/services/face_crop.py`)
- `app/services/image_ops.py` — resize/preprocess แบบ NumPy ที่ให้ค่าเท่ากับ TensorFlow ตอนเทรน (convert_tflite.py ตรวจให้)
- `face_landmarker.task` — โมเดลหาใบหน้าของ MediaPipe (~3.6 MB) ดาวน์โหลดอัตโนมัติครั้งแรก ต้องต่ออินเทอร์เน็ต
- โหลดโมเดลเบื้องหลังตอนเปิดเซิร์ฟเวอร์ — log `Face-shape model ready` = พร้อมใช้
- ปรับตามเครื่อง (Render → Environment):

| ตัวแปร | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `FACE_SHAPE_MAX_MODELS` | `2` | เก็บโมเดลกี่เพศไว้ใน RAM พร้อมกัน (1 เพศ = ทั้ง 3 fold) — RAM ไม่พอให้ตั้ง `1` |
| `FACE_SHAPE_ENSEMBLE` | ทั้งหมด | ใช้แค่ N fold แรก — ถ้าเซิร์ฟเวอร์ช้าเกินไปตั้ง `1` |
| `FACE_SHAPE_TTA` | ตาม `class_names.json` | `0` = ปิดการทายภาพกลับซ้าย-ขวา (เร็วขึ้นเท่าตัว) |
| `FACE_SHAPE_THREADS` | `1` | จำนวน thread ต่อโมเดล |

- ไม่มีโมเดลในเครื่อง dev? ตั้ง `FACE_SHAPE_PREDICTOR=mock` เพื่อใช้ผลสุ่มชั่วคราว
- เปลี่ยนโมเดลใหม่: รัน `python scripts/split_model.py <ไฟล์ .tflite ...>` แล้ววาง `class_names.json` ใหม่จาก Kaggle
- โมเดลเดิม (`face_shape_effnetb1.keras`, `.keras` แบบ fold เดียว) ยังใช้ได้ถ้าติดตั้ง TensorFlow เอง แต่เซิร์ฟเวอร์ไม่ติดตั้งแล้ว

## สร้างภาพทรงผมสมจริงด้วย AI (`/hair/realistic`)

ในหน้าลองทรงผม (ภาพนิ่ง) สมาชิกกด **"สร้างภาพสมจริง"** ได้วันละ `HAIR_AI_DAILY_LIMIT` รูป (ค่าเริ่มต้น 5, นับเฉพาะครั้งที่สำเร็จ, รีเซ็ตเที่ยงคืนเวลาไทย)
ปุ่มจะซ่อนเองถ้ายังไม่ได้ใส่ key ใด ๆ — รูปจะถูกส่งไปที่บริการภายนอก (หน้าเว็บขอความยินยอมก่อนครั้งแรก) แต่เซิร์ฟเวอร์ไม่เก็บรูป เก็บแค่จำนวนครั้ง (ตาราง `hair_generations`)

| ตัวเลือก | ตั้งค่า (Render → Environment) | วิธีทำงาน |
|---|---|---|
| A. Cloudflare Workers AI (ฟรี) | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | หน้าเว็บส่งภาพลองผมแบบ 2D ครอปรอบศีรษะ 512×512 + mask บริเวณผม → Stable Diffusion 1.5 inpainting วาดเส้นผมใหม่เฉพาะในบริเวณผม แล้วหน้าเว็บนำกลับไปวางบนรูปเต็ม (ใบหน้าเป็นของเดิม 100%) |
| B. Google Gemini (คุณภาพสูง) | `GEMINI_API_KEY` (+ `HAIR_AI_PROVIDER=gemini` ถ้าใส่ทั้งสองแบบ) | ส่งรูปต้นฉบับ + รูปทรงผมที่เลือกเป็นตัวอย่าง → Gemini แก้เฉพาะผมแล้วส่งรูปทั้งรูปกลับมา |

**สมัคร Cloudflare (ฟรี ไม่ต้องผูกบัตร):** dash.cloudflare.com → สมัคร/เข้าสู่ระบบ → เมนู **AI → Workers AI** → **Use REST API** → กด **Create a Workers AI API Token** แล้วคัดลอก token และ **Account ID** ที่หน้าเดียวกัน

**สมัคร Gemini:** aistudio.google.com → **Get API key** → สร้าง key ในโปรเจกต์ Google Cloud ที่เปิด billing แล้ว (บัญชี Google Cloud ใหม่ได้เครดิตทดลองใช้ $300 / 90 วัน)

สลับผู้ให้บริการได้ทันทีด้วย `HAIR_AI_PROVIDER=cloudflare` หรือ `gemini` แล้ว Deploy ใหม่ — ปิดฟีเจอร์ด้วย `HAIR_AI_PROVIDER=off`
