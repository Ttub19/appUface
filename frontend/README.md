# Face Shape AI — Frontend

Web app: 1) วิเคราะห์รูปทรงใบหน้า 2) วิเคราะห์ Golden Ratio แบบเรียลไทม์ 3) ลองแว่นตาเสมือนจริง 4) ลองทรงผมเสมือนจริง — ติดตามใบหน้าด้วย **MediaPipe Face Mesh** (468 landmarks) ผ่านกล้องเว็บแคม ทั้งหมดทำงานฝั่ง client

ส่วน **auth, บันทึกประวัติ/รายการโปรด/ผล Golden Ratio, และ Admin stats เรียก backend จริง** (ดู `../backend/` และ `../README.md` สำหรับวิธีรันทั้งระบบ)

## เทคโนโลยีที่ใช้
- HTML5 + Tailwind CSS (CDN) + Vanilla JavaScript (ES6+ Modules)
- MediaPipe Face Mesh + Camera Utils (CDN) สำหรับติดตามจุด Landmark บนใบหน้า (ทำงานฝั่ง client ล้วน ๆ ไม่ผ่าน backend)
- Chart.js (CDN) สำหรับกราฟในหน้า Admin (ข้อมูลจริงจาก `GET /admin/stats`)
- ไม่มี build step

## โครงสร้างโปรเจกต์ (Clean Architecture)

```
frontend/
├── index.html                 # markup หลัก ทุกหน้าจอ (screens) ของแอป
├── Dockerfile + nginx.conf     # เสิร์ฟไฟล์ static + reverse proxy /api/* ไปที่ backend
├── package.json                # สคริปต์รัน dev server แบบ standalone (ไม่ต้อง npm install)
├── assets/hair/*.svg           # ทรงผมเวกเตอร์ที่วาดเอง (เทมเพลต 400×560: hairline anchor (200,168), ระยะขมับ 200px)
├── css/style.css
└── js/
    ├── app.js                  # entry point: ผูกทุกโมดูลเข้าด้วยกัน + expose ฟังก์ชันให้ inline onclick ใน HTML + restore session ตอนโหลดหน้า
    ├── apiClient.js             # fetch wrapper เรียก backend จริงทุก endpoint + จัดการ JWT ใน localStorage
    ├── state.js                 # shared mutable state (ผู้ใช้ปัจจุบัน, รูปที่เลือก, ผลวิเคราะห์, cache รายการโปรด ฯลฯ)
    ├── data.js                  # ข้อมูลสไตล์แนะนำ (hair/glasses/jewelry tips) ต่อรูปทรงหน้า 5 แบบ — เนื้อหาการนำเสนอ ไม่ใช่ผลจาก backend
    ├── landmarks.js              # ดัชนีจุด Landmark ของ MediaPipe ที่ใช้ในแอป + helper clamp()
    ├── cameraUtils.js            # FaceARSession — เปิด/ปิดกล้อง + รัน MediaPipe FaceMesh ต่อเฟรม
    ├── arRenderer.js             # แปลง landmark เป็น transform (ตำแหน่ง/หมุน/สเกล) + รูปแว่น/ทรงผม (SVG) + วาด mesh
    ├── goldenRatioCalc.js        # คำนวณสัดส่วน Golden Ratio จาก Euclidean distance ของ landmark จริง (ฝั่ง client)
    ├── hairTryOn/                # ลองทรงผมเรียลไทม์ (MediaPipe Tasks Vision: Face Landmarker + Hair Segmenter)
    │   ├── hairTryOnEngine.js    # กล้อง + โหลดโมเดล (GPU→CPU fallback) + วาดเฟรม/กลบผมจริง/ซ้อนทรงใหม่ลง canvas เดียว
    │   ├── headPose.js           # แปลง facial transformation matrix → yaw/pitch/roll + คำนวณตำแหน่ง/สเกลทรงผม (pure, มี test)
    │   ├── hairEraser.js         # สร้างเลเยอร์กลบผมจริงจาก hair mask (สีผิว/สีพื้นหลัง) (pure, มี test)
    │   └── hairAssets.js         # แคตตาล็อกทรงผม + หมวด face shape ที่แนะนำ
    └── ui/
        ├── toast.js              # แจ้งเตือนแบบ toast
        ├── navigation.js         # สลับหน้าจอ + ปิดกล้องทุกตัว + เรียกโหลดข้อมูลของแต่ละหน้าจากหลังบ้านตอนเข้าหน้านั้น
        ├── auth.js                # login/register/guest/logout จริงผ่าน JWT + สิทธิ์ตามบทบาท (guest/member/admin)
        ├── scanScreen.js          # อัปโหลด/ถ่ายภาพ + เรียก POST /scans/predict จริง + toggle รายการโปรด
        ├── goldenScreen.js        # หน้าวิเคราะห์ Golden Ratio (เปิดกล้อง, วาด mesh, อัปเดตคะแนน, บันทึกผลผ่าน POST /golden-ratio)
        ├── glassesScreen.js       # หน้าลองแว่นเสมือนจริง (AR ติดตามตา — ไม่เกี่ยวกับ backend)
        ├── hairScreen.js          # หน้าลองทรงผมเสมือนจริง: กรองทรงตามรูปหน้า, เปิด/ปิดกลบผมจริง, ถ่ายรูป (ไม่เกี่ยวกับ backend)
        ├── guideScreen.js         # คลังความรู้ทรงหน้า (ข้อมูล static)
        ├── favoritesScreen.js     # รายการโปรด — CRUD ผ่าน backend จริง
        ├── historyScreen.js       # ประวัติการสแกน — CRUD + export CSV ผ่าน backend จริง
        ├── profileScreen.js       # โปรไฟล์ + ธีมสี — โหลด/บันทึกผ่าน backend จริง
        └── adminScreen.js         # กราฟแดชบอร์ด Admin — ข้อมูลจริงจาก GET /admin/stats
```

## วิธีรัน

### แนะนำ: รันทั้งระบบจาก root ด้วย Docker Compose

```bash
cd ..   # กลับไปที่ root ของโปรเจกต์ (face-shape-app/)
docker compose up --build
```

ดูรายละเอียดเต็มใน `../README.md` — วิธีนี้ nginx จะ reverse-proxy `/api/*` ไปที่ backend ให้อัตโนมัติ ไม่ต้องตั้งค่าอะไรเพิ่ม

### แบบ standalone (รัน frontend เดี่ยว ๆ ต่อกับ backend ที่รันอยู่คนละที่)

```bash
npm start
# หรือ: python3 -m http.server 5173
```

ต้องมี backend รันอยู่แล้ว (ดู `../backend/README.md`) และต้องบอก frontend ว่า backend อยู่ที่ไหน เพราะจะไม่มี nginx proxy คอย forward `/api` ให้ — เพิ่ม `<script>` นี้ใน `index.html` ก่อนแท็ก `<script type="module" src="js/app.js">`:

```html
<script>window.__API_BASE_URL__ = "http://localhost:8000";</script>
```

และตั้ง `CORS_ORIGINS` ใน backend's `.env` ให้รวม origin ของ frontend (เช่น `http://localhost:5173`) ด้วย

## ฟีเจอร์หลัก
- **สแกน & วิเคราะห์**: อัปโหลด/ถ่ายภาพ → เรียก backend จริงวิเคราะห์รูปทรงใบหน้า พร้อม loading/success/error state ที่สมบูรณ์
- **วิเคราะห์ Golden Ratio**: คำนวณสัดส่วนสามส่วนแนวตั้ง, อัตราส่วนยาว:กว้างเทียบ Φ (1.618), คะแนนความสมมาตร จาก landmark จริง 468 จุด แบบเรียลไทม์ (ฝั่ง client) พร้อมบันทึกผลลง PostgreSQL ได้
- **ลองแว่นเสมือนจริง**: แว่นหมุน/ย่อ-ขยายตามตำแหน่งดวงตาที่ติดตามแบบเรียลไทม์
- **ลองทรงผมเสมือนจริง**: ทรงผมหมุน/ย่อ-ขยายตามตำแหน่ง-มุมของศีรษะที่ติดตามแบบเรียลไทม์
- คลังความรู้ทรงหน้า, รายการโปรด, ประวัติการสแกน (ค้นหา/ลบ/export CSV), Admin dashboard — ทั้งหมดอ่าน-เขียนจริงผ่าน backend + PostgreSQL

## หมายเหตุด้านความปลอดภัย
- ไม่มี hardcoded secret/API key ในโค้ด — `window.__API_BASE_URL__` เป็น config ไม่ใช่ secret
- ทุกฟีเจอร์ที่ใช้กล้องมี try/catch รอบการขอสิทธิ์กล้องและรอบการเรียก MediaPipe ทุกจุด พร้อมข้อความแจ้งผู้ใช้เมื่อผิดพลาด
- ทุกการเรียก backend มี try/catch + toast แจ้ง error เป็นภาษาไทย
- JWT เก็บใน `localStorage` (มาตรฐานทั่วไปสำหรับเว็บแอป — ไม่ใช่ Claude.ai Artifact ที่ห้ามใช้ browser storage)
